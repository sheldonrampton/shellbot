"""
chatbotter.py:
Generates embeddings from the GEM wiki, saves the embeddings in Pinecone,
and saves the article segments in postgres.

For more information see:
https://cookbook.openai.com/examples/embedding_wikipedia_articles_for_search
"""

# imports
from openai import OpenAI # for calling the OpenAI API
import mwclient  # for downloading example Wikipedia articles
from typing import List, Iterator
from mediawiki import MediaWiki
import mwparserfromhell  # for splitting Wikipedia articles into sections
import os  # for environment variables
import pandas as pd  # for DataFrames to store article sections and embeddings
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens
from pinecone import Pinecone, ServerlessSpec
import hashlib
import time
import numpy as np
import itertools
import ast  # for converting embeddings saved as strings back to arrays
from scipy import spatial  # for calculating vector similarities for search
import json
import psycopg2
from psycopg2 import sql
import pickle
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import requests


#### HELPER FUNCTIONS ###
# Format a JSON string so it is easy to read.
def show_json(obj):
    print(json.dumps(json.loads(obj.model_dump_json()), indent=2))


# Pretty printing helper
def pretty_print(messages):
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
    print()


# Models a simple batch generator that make chunks out of an input DataFrame
class WikiExtractor:
    def __init__(self,
        site_name: str = "www.gem.wiki",
        url: str = 'https://www.gem.wiki/w/api.php',
        user_agent: str = 'sheldon-ramptons-agent',
        gpt_model: str = "gpt-4o",  # selects which tokenizer to use
        limit = False,
        debug = False,
        pickle_file = None
    ) -> None:
        self.site_name = site_name
        self.site = mwclient.Site(site_name)
        self.gw = MediaWiki(url=url, user_agent=user_agent)
        self.url = url
        self.user_agent = user_agent
        self.gpt_model = gpt_model
        self.limit = limit
        self.sections_to_ignore = [
            "See also",
            "References",
            "External links",
            "Further reading",
            "Footnotes",
            "Bibliography",
            "Sources",
            "Citations",
            "Literature",
            "Footnotes",
            "Notes and references",
            "Photo gallery",
            "Works cited",
            "Photos",
            "Gallery",
            "Notes",
            "References and sources",
            "References and notes",
        ]
        self.debug = debug
        self.pickle_file = pickle_file
    
    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5)
    )
    def query_wiki(self, query):
        result = self.gw.opensearch(query, results=1)
        if result and len(result[0]) >= 3:
            # Only assign if there is a valid result with at least 3 elements
            return result[0][2]
        else:
            print(f"Had trouble getting the url for {query}")
            return 'https://' + self.site_name + '/' + query.replace(" ", "_")

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5)
    )
    def page_text(self, title):
        page = self.site.pages[title]
        return page.text()

    # Makes chunks out of an input DataFrame
    def list_of_titles(
        self
    ) -> set[str]:
        """Return a set of page titles in a given Wiki category and its subcategories."""
        urls = {}
        pages = self.site.allpages()
        titles = set()
        i = 0
        if self.limit:
            pages = itertools.islice(pages, self.limit)
        for page in pages:
            title = page.name
            titles.add(title)
            try:
                # Example query to the MediaWiki
                urls[title] = self.query_wiki(title)
            except requests.exceptions.RequestException as e:
                print(f"Failed to get URL for {title} after multiple retries: {e}")
                urls[title] = 'https://' + self.site_name + '/' + title.replace(" ", "_")
            if i % 100 == 0:
                print(title + ": " + urls[title])
            i += 1
        return titles, urls

    def titles_from_category(
        self,
        category: mwclient.listing.Category,
        category_names,
        max_depth: int
    ) -> set[str]:
        """Return a set of page titles in a given Wiki category and its subcategories."""
        titles = set()
        try:
            # Example query to the MediaWiki
            category_members = category.members()
        except:
            print(f"Failed to get members for {category}")
            return(titles, category_names)
        for cm in category_members:
            if type(cm) == mwclient.page.Page:
                # ^type() used instead of isinstance() to catch match w/ no inheritance
                titles.add(cm.name)
            elif isinstance(cm, mwclient.listing.Category) and max_depth > 0:
                category_names.add(cm.name)
                print("Category", str(max_depth) + ": " + cm.name)
                deeper_titles, category_names = self.titles_from_category(cm, category_names = category_names, max_depth=max_depth - 1)
                titles.update(deeper_titles)
        return titles, category_names

    def all_subsections_from_section(
        self, section: mwparserfromhell.wikicode.Wikicode,
        parent_titles: list[str],
    ) -> list[tuple[list[str], str]]:
        """
        From a Wikipedia section, return a flattened list of all nested subsections.
        Each subsection is a tuple, where:
            - the first element is a list of parent subtitles, starting with the page title
            - the second element is the text of the subsection (but not any children)
        """
        headings = [str(h) for h in section.filter_headings()]
        title = headings[0]
        if title.strip("=" + " ") in self.sections_to_ignore:
            # ^wiki headings are wrapped like "== Heading =="
            return []
        titles = parent_titles + [title]
        full_text = str(section)
        section_text = full_text.split(title)[1]
        if len(headings) == 1:
            return [(titles, section_text)]
        else:
            first_subtitle = headings[1]
            section_text = section_text.split(first_subtitle)[0]
            results = [(titles, section_text)]
            for subsection in section.get_sections(levels=[len(titles) + 1]):
                results.extend(self.all_subsections_from_section(subsection, titles))
            return results

    def all_subsections_from_title(
        self,
        title: str,
    ) -> list[tuple[list[str], str]]:
        """From a Wikipedia page title, return a flattened list of all nested subsections.
        Each subsection is a tuple, where:
            - the first element is a list of parent subtitles, starting with the page title
            - the second element is the text of the subsection (but not any children)
        """
        try:
            text = self.page_text(title)
        except:
            print(f"Failed to get full text for {title} after multiple retries")
            return [([title], title)]
        parsed_text = mwparserfromhell.parse(text)
        headings = [str(h) for h in parsed_text.filter_headings()]
        if headings:
            summary_text = str(parsed_text).split(headings[0])[0]
        else:
            summary_text = str(parsed_text)
        results = [([title], summary_text)]
        for subsection in parsed_text.get_sections(levels=[2]):
            results.extend(self.all_subsections_from_section(subsection, [title]))
        return results

    # clean text
    def clean_section(self, section: tuple[list[str], str]) -> tuple[list[str], str]:
        """
        Return a cleaned up section with:
            - <ref>xyz</ref> patterns removed
            - leading/trailing whitespace removed
        """
        titles, text = section
        text = re.sub(r"<ref.*?</ref>", "", text)
        text = text.strip()
        return (titles, text)

    # filter out short/blank sections
    def keep_section(self, section: tuple[list[str], str]) -> bool:
        """Return True if the section should be kept, False otherwise."""
        titles, text = section
        if len(text) < 16:
            return False
        else:
            return True

    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(self.gpt_model)
        return len(encoding.encode(text))

    def halved_by_delimiter(self, string: str, delimiter: str = "\n") -> list[str, str]:
        """Split a string in two, on a delimiter, trying to balance tokens on each side."""
        chunks = string.split(delimiter)
        if len(chunks) == 1:
            return [string, ""]  # no delimiter found
        elif len(chunks) == 2:
            return chunks  # no need to search for halfway point
        else:
            total_tokens = self.num_tokens(string)
            halfway = total_tokens // 2
            best_diff = halfway
            for i, chunk in enumerate(chunks):
                left = delimiter.join(chunks[: i + 1])
                left_tokens = self.num_tokens(left)
                diff = abs(halfway - left_tokens)
                if diff >= best_diff:
                    break
                else:
                    best_diff = diff
            left = delimiter.join(chunks[:i])
            right = delimiter.join(chunks[i:])
            return [left, right]

    def truncated_string(
        self,
        string: str,
        max_tokens: int,
        print_warning: bool = True,
    ) -> str:
        """Truncate a string to a maximum number of tokens."""
        encoding = tiktoken.encoding_for_model(self.gpt_model)
        encoded_string = encoding.encode(string)
        truncated_string = encoding.decode(encoded_string[:max_tokens])
        if print_warning and len(encoded_string) > max_tokens:
            first_line = string.split('\n')[0]
            print(f"Warning: Truncated string {first_line} from {len(encoded_string)} tokens to {max_tokens} tokens.")
        return truncated_string

    def split_strings_from_subsection(
        self,
        subsection: tuple[list[str], str],
        max_tokens: int = 10000,
        max_recursion: int = 5,
    ) -> list[str]:
        """
        Split a subsection into a list of subsections, each with no more than max_tokens.
        Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
        """
        titles, text = subsection
        string = "\n\n".join(titles + [text])
        num_tokens_in_string = self.num_tokens(string)
        # if length is fine, return string
        if num_tokens_in_string <= max_tokens:
            return [string]
        # if recursion hasn't found a split after X iterations, just truncate
        elif max_recursion == 0:
            return [self.truncated_string(string, max_tokens=max_tokens)]
        # otherwise, split in half and recurse
        else:
            titles, text = subsection
            for delimiter in ["\n\n", "\n", ". "]:
                left, right = self.halved_by_delimiter(text, delimiter=delimiter)
                if left == "" or right == "":
                    # if either half is empty, retry with a more fine-grained delimiter
                    continue
                else:
                    # recurse on each half
                    results = []
                    for half in [left, right]:
                        half_subsection = (titles, half)
                        half_strings = self.split_strings_from_subsection(
                            half_subsection,
                            max_tokens=max_tokens,
                            max_recursion=max_recursion - 1,
                        )
                        results.extend(half_strings)
                    return results
        # otherwise no split was found, so just truncate (should be very rare)
        return [self.truncated_string(string, max_tokens=max_tokens)]

    def compile_titles(
        self,
        categories = None,
        max_depth: int = 1
    ):
        if isinstance(categories, str):
            categories = [categories]
        if categories:
            all_titles = set()
            for category_title in categories:
                full_category_title = "Category:" + category_title
                category_page = self.site.pages[full_category_title]
                titles, category_names = self.titles_from_category(category = category_page, category_names = set(), max_depth=max_depth)
                all_titles = all_titles.union(titles)
            titles = all_titles
            if self.limit:
                titles = set(list(titles)[:self.limit])
            urls = {}
            i = 0
            for title in titles:
                if i % 100 == 0:
                    print(f"{i}: {title}")
                i += 1
                try:
                    # Example query to the MediaWiki
                    urls[title] = self.query_wiki(title)
                except requests.exceptions.RequestException as e:
                    print(f"Failed to get URL for {title} after multiple retries: {e}")
                    urls[title] = 'https://' + self.site_name + '/' + title.replace(" ", "_")

        else:
            titles, urls = self.list_of_titles()
            category_names = set()

        if self.pickle_file is not None:
                # Example data: A set containing a list of strings, a dictionary, and another set containing URLs
            data = ({
                "titles": titles,
                "urls": urls,
                "category_names": category_names
            })
            # Saving the data to a file
            with open(self.pickle_file, "wb") as f:
                pickle.dump(data, f)
        return titles, urls, category_names

    def compile_wiki_strings(
        self,
        categories = None,
        max_depth: int = 1
    ):
        if self.pickle_file is not None and os.path.exists(self.pickle_file):
            # File exists, load data from the file using pickle
            print(f"FOUND {self.pickle_file}")
            with open(self.pickle_file, "rb") as f:
                data = pickle.load(f)
                titles = data["titles"]
                urls = data["urls"]
                category_names =  data["titles"]
                print("Loaded data from file.")
        else:
            print(f"DIDN'T FIND {self.pickle_file}")
            titles, urls, category_names = self.compile_titles(categories = categories, max_depth = max_depth)
            print("Generated data and saved to file.")
        # split pages into sections
        # may take ~1 minute per 100 articles
        wiki_sections = []
        i = 0
        for title in titles:
            wiki_sections.extend(self.all_subsections_from_title(title))
            if i % 100 == 0:
                print(title)
            i += 1
        if self.debug:
            print(f"Found {len(wiki_sections)} sections in {len(titles)} pages.")
        wiki_sections = [self.clean_section(ws) for ws in wiki_sections]
        original_num_sections = len(wiki_sections)
        wiki_sections = [ws for ws in wiki_sections if self.keep_section(ws)]
        if self.debug:
            print(f"Filtered out {original_num_sections-len(wiki_sections)} sections, leaving {len(wiki_sections)} sections.")
            # print example data
            for ws in wiki_sections[:5]:
                print(ws[0])
                print(ws[1][:77] + "...")
                print()
        # split sections into chunks
        MAX_TOKENS = 1600
        strings = []
        for section in wiki_sections:
            strings.extend(self.split_strings_from_subsection(section, max_tokens=MAX_TOKENS))

        if self.debug:
            print(f"{len(wiki_sections)} Wikipedia sections split into {len(strings)} strings.")
            # print example data
            print(strings[1])
        return strings, urls

    def chunk_file(self, file_path, title=None, max_tokens=1000):
        if title is None:
            title = file_path
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            # If UTF-8 fails, fall back to ISO-8859-1 (or another encoding)
            with open(file_path, 'r', encoding='ISO-8859-1') as file:
                text = file.read()
        return self.split_strings_from_subsection(
            [[title], text], max_tokens = max_tokens
        )


class Embedder:
    def __init__(
        self,
        openai_client,
        batch_size = 1000,
        embedding_model = "text-embedding-3-small",
        gpt_model: str = "gpt-4o",  # selects which tokenizer to use
        debug = False
    ) -> None:
        self.openai_client = openai_client
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.gpt_model = gpt_model
        self.debug = debug
        self.urls = {}

    def get_first_line(self, text):
        return text.split('\n')[0]

    def get_url(self, title):
        if title in self.urls.keys():
            return self.urls[title]
        else:
            return ''

    def generate_vector_id(self, text: str) -> str:
        # Create a SHA256 hash object
        hash_object = hashlib.sha256()
        
        # Update the hash object with the text encoded in UTF-8
        hash_object.update(text.encode('utf-8'))
        
        # Return the hexadecimal digest of the hash, which is a string representation of the hash
        return hash_object.hexdigest()

    def compile_embeddings(self, strings, urls):
        embeddings = []
        self.urls = urls
        for batch_start in range(0, len(strings), self.batch_size):
            print(f"Embedding a batch of {self.batch_size} strings...")
            batch_end = batch_start + self.batch_size
            batch = strings[batch_start:batch_end]
            if self.debug:
                print(f"Batch {batch_start} to {batch_end-1}")
            response = self.openai_client.embeddings.create(model=self.embedding_model, input=batch)
            for i, be in enumerate(response.data):
                assert i == be.index  # double check embeddings are in same order as input
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)

        df = pd.DataFrame({"text": strings, "embedding": embeddings})
        df["title"] = df['text'].apply(self.get_first_line)
        df["url"] = df['title'].apply(self.get_url)
        df["vector_id"] = df.apply(lambda row: self.generate_vector_id(row['url'] + row['text']), axis=1)

        if self.debug:
            for value in df['title']:
                print(value)
            for value in df['url']:
                print(value)
            for value in df['vector_id']:
                print(value)

        return df


# Models a simple batch generator that makes chunks out of an input DataFrame
class BatchGenerator:
    def __init__(self, batch_size: int = 10) -> None:
        self.batch_size = batch_size
    
    # Makes chunks out of an input DataFrame
    def to_batches(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        splits = self.splits_num(df.shape[0])
        if splits <= 1:
            yield df
        else:
            for chunk in np.array_split(df, splits):
                yield chunk

    # Determines how many chunks DataFrame contains
    def splits_num(self, elements: int) -> int:
        return round(elements / self.batch_size)
    
    __call__ = to_batches


class Storer:
    def __init__(
        self,
        openai_client,
        df = None,
        knowledge_db_name: str = 'gem_wiki_50_knowledge',
        logs_database_name = None,
        logs_user = None,
        logs_password = None,
        db_host = None,
        db_port = None,
        pinecone_index_name = 'gem-wiki-50',
        embedding_model = "text-embedding-3-small",
        overwrite_db = False,
        overwrite_pinecone = False,
        debug = False
    ) -> None:
        self.openai_client = openai_client
        self.df = df
        self.db_host = db_host if db_host is not None else os.getenv('DB_HOST')
        self.db_port = db_port if db_port is not None else os.getenv('DB_PORT')
        self.knowledge_db_name = knowledge_db_name
        self.logs_database_name = logs_database_name if logs_database_name is not None else os.getenv('SHELLBOT_DB_NAME')
        self.logs_user = logs_user if logs_user is not None else os.getenv('SHELLBOT_USER')
        self.logs_password = logs_password if logs_password is not None else os.getenv('SHELLBOT_USER_PASSWORD')
        self.pinecone_index_name = pinecone_index_name
        self.embedding_model = embedding_model
        self.overwrite_db = overwrite_db
        self.overwrite_pinecone = overwrite_pinecone
        self.debug = debug
        self.setup_database()
        self.setup_pinecone()
        if df is not None:
            self.upsert_data()

    def database_connection(self):
        conn = psycopg2.connect(
            dbname=self.logs_database_name,
            user=self.logs_user,
            password=self.logs_password,
            host=self.db_host,
            port=self.db_port,
            sslmode='allow'
        )
        cur = conn.cursor()
        return conn, cur

    def create_table(self, table_name, create_table_query):
        conn, cur = self.database_connection()
        cur.execute(create_table_query)
        cur.execute(sql.SQL("GRANT ALL PRIVILEGES ON TABLE {} TO {}").format(
            sql.Identifier(table_name),
            sql.Identifier(self.logs_user)
        ))
        conn.commit()
        cur.close()
        conn.close()

    def setup_database(self):
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.knowledge_db_name} (
            unique_id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            url TEXT
        );
        """
        self.create_table(self.knowledge_db_name, create_table_query)

    def setup_pinecone(self):
        # Check whether the index with the same name already exists - if so, delete it
        pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        pinecone = Pinecone(api_key=pinecone_api_key)
        if self.pinecone_index_name in pinecone.list_indexes() and self.overwrite_pinecone:
            pinecone.delete_index(self.pinecone_index_name)

        # Creates index if it doesn't already exist.
        if self.pinecone_index_name not in pinecone.list_indexes().names():
            # if does not exist, create index
            spec = ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
            # Calculate length of embedding based on the embedding model.
            response = self.openai_client.embeddings.create(
              input="Hello!",
              model=self.embedding_model
            )
            embedding_length = len(response.data[0].embedding)
            pinecone.create_index(
                self.pinecone_index_name,
                dimension=embedding_length,
                metric='cosine',
                spec=spec
            )
            # wait for index to be initialized
            while not pinecone.describe_index(self.pinecone_index_name).status['ready']:
                time.sleep(1)

        # connect to index
        self.pinecone_index = pinecone.Index(self.pinecone_index_name)
        time.sleep(1)
        if self.debug:
            # view index stats
            print(self.pinecone_index.describe_index_stats())
            # Confirm our index was created
            print(pinecone.list_indexes())

    def upsert_data(self):
        # Upsert content vectors in content namespace - this can take a few minutes
        i = 0
        if self.debug:
            print("Uploading vectors to content namespace..")
        conn, cur = self.database_connection()
        df_batcher = BatchGenerator(200)
        for batch_df in df_batcher(self.df):
            self.pinecone_index.upsert(vectors=zip(
                batch_df.vector_id, batch_df.embedding,
                [{**a, **b} for a, b in zip(
                    [{ "title": t } for t in batch_df.title ],
                    [{ "url": u } for u in batch_df.url ])
                ]
            ), namespace='content')
            for rownum, row in batch_df.iterrows():
                if i % 100 == 0:
                    print("Upserting " + row['title'])
                i += 1
                try:
                    cur.execute(f'''
                    INSERT INTO {self.knowledge_db_name} (unique_id, title, content, url)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (unique_id) DO NOTHING
                    ''', (row['vector_id'], row['title'], row['text'], row['url']))
                    if self.debug:
                        print("Inserted row ", rownum, row['vector_id'], row['title'])
                except psycopg2.Error as e:
                    print(f"An error occurred: {e}")
                    print(row)
        conn.commit()
        conn.close()
        if self.debug:
            print("Records inserted successfully.")

            # Check index size for each namespace to confirm all of our docs have loaded
            print(self.pinecone_index.describe_index_stats())

    def get_article_chunk(self, unique_id):
        conn, cur = self.database_connection()
        try:
            # SQL query to retrieve the row with the specified unique_id
            query = f"SELECT title, url, content FROM {self.knowledge_db_name} WHERE unique_id = %s"
            cur.execute(query, (unique_id, ))
            row = cur.fetchone()
            conn.close()

            # Check if a row was found
            if row:
                return row
            else:
                print(f"No row found with unique_id = {unique_id}")
                return ['', '', '']
        except psycopg2.Error as e:
            print(f"An error occurred: {e}")
            return ['', '', '']

    # search function
    def get_pinecone_matches(
        self,
        query: str,
        top_n: int = 100
    ) -> tuple[list[str], list[float]]:
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query,
        )
        query_embedding = query_embedding_response.data[0].embedding

        res = self.openai_client.embeddings.create(input=[query], model=self.embedding_model)
        embedded_query = res.data[0].embedding

        # Query namespace passed as parameter using title vector
        query_result = self.pinecone_index.query(
            namespace='content',
            vector=embedded_query,
            top_k=top_n
        )

        if self.debug:
            print(f'\nMost similar results to {query} in "content" namespace:\n')
            if not query_result.matches:
                print('no query result')
        
        matches = query_result.matches
        ids = [res.id for res in matches]
        scores = [res.score for res in matches]
        df = pd.DataFrame({'id':ids, 
                           'score':scores,
                           })
        
        df['title'], df['url'], df['content'] = zip(*df['id'].apply(lambda x: self.get_article_chunk(x)))
        return df

    def query_article(self, query, namespace, top_k=5):
        '''Queries an article using its title in the specified
         namespace and prints results.'''

        # Use the OpenAI client to create vector embeddings based on the title column
        res = self.openai_client.embeddings.create(input=[query], model=self.embedding_model)
        embedded_query = res.data[0].embedding

        # Query namespace passed as parameter using title vector
        query_result = self.pinecone_index.query(
            namespace=namespace,
            vector=embedded_query,
            top_k=top_k
        )

        # Print query results 
        if self.debug:
            print(f'\nMost similar results to {query} in "{namespace}" namespace:\n')
            if not query_result.matches:
                print('no query result')
        
        matches = query_result.matches
        ids = [res.id for res in matches]
        scores = [res.score for res in matches]
        df = pd.DataFrame({'id':ids, 
                           'score':scores
                           })
        df['title'], df['url'], df['content'] = zip(*df['id'].apply(lambda x: self.get_article_chunk(x)))
        
        if self.debug:
            counter = 0
            for k,v in df.iterrows():
                counter += 1
                print(f'{v.title} (score = {v.score})')
            
            print('\n')

        return df


class Asker:
    def __init__(
        self,
        openai_client,
        df: pd.DataFrame = None,
        storage = None,
        embedding_model = "text-embedding-3-small",
        gpt_model: str = "gpt-4o",  # selects which tokenizer to use
        introduction: str = 'Use the below articles from the Global Energy Monitor wiki to answer questions. If the answer cannot be found in the articles, write "I could not find an answer."',
        string_divider: str = 'Global Energy Monitor section:',
        debug = False
    ) -> None:
        self.openai_client = openai_client
        self.embedding_model = embedding_model
        self.gpt_model = gpt_model
        self.debug = debug
        self.introduction = introduction
        self.string_divider = string_divider
        self.df = df
        self.storage = storage

    def load_embeddings_from_csv(self, embeddings_path):
        df = pd.read_csv(embeddings_path)
        # convert embeddings from CSV str type back to list type
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
        # the dataframe has two columns: "text" and "embedding"
        if self.debug:
            print(df)
        self.df = df
        return df

    # search function
    def strings_ranked_by_relatedness(
        # # examples of strings ranked by relatedness
        # strings, relatednesses = strings_ranked_by_relatedness("Wisconsin", top_n=5)
        # for string, relatedness in zip(strings, relatednesses):
        #     print(f"{relatedness=:.3f}")
        #     print(string)
        self,
        query: str,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
    ) -> tuple[list[str], list[float]]:
        """Returns a list of strings and relatednesses, sorted from most related to least."""
        query_embedding_response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query,
        )
        query_embedding = query_embedding_response.data[0].embedding
        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in self.df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]

    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(self.gpt_model)
        return len(encoding.encode(text))

    def query_message(
        self,
        query: str,
        token_budget: int,
        storage = None
    ) -> str:
        """Return a message for GPT, with relevant source texts pulled from a dataframe."""
        articles = {}
        if self.storage:
            df = self.storage.get_pinecone_matches(query)
            question = f"\n\nQuestion: {query}"
            message = self.introduction
            for k,v in df.iterrows():
                string = v.content
                title = v.title
                url = v.url
                articles[title] = url
                next_article = f'\n\n{self.string_divider}\n"""\n{string}\n"""'
                if (
                    self.num_tokens(message + next_article + question)
                    > token_budget
                ):
                    break
                else:
                    message += next_article
        else:
            strings, relatednesses = self.strings_ranked_by_relatedness(query)
            question = f"\n\nQuestion: {query}"
            message = self.introduction
            for string in strings:
                next_article = f'\n\n{self.string_divider}\n"""\n{string}\n"""'
                if (
                    self.num_tokens(message + next_article + question)
                    > token_budget
                ):
                    break
                else:
                    message += next_article
        return message + question, articles

    def add_to_history(self, conversation_history, role, content):
        conversation_history.append({"role": role, "content": content})

    def build_contextual_input(self, conversation_history, user_query):
        # Concatenate previous messages and the current query
        context = ""
        for message in conversation_history[-10:]:  # Use the last 10 messages for context
            context += f"{message['role']}: {message['content']}\n"
        context += f"user: {user_query}\n"
        return context

    def ask(
        self,
        query,
        model = None,
        conversation_history = [],
        token_budget: int = 4096 - 500
    ):
        """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
        if not model:
            model = self.gpt_model
        if len(conversation_history) > 0:
            self.add_to_history(conversation_history, "user", query)
            if self.debug:
                for m in conversation_history:
                    print(m)
            contextual_input = self.build_contextual_input(conversation_history, query)
            message, articles = self.query_message(contextual_input, token_budget=token_budget)
            if message == "I could not find an answer.":
                return self.ask(query, model, [], token_budget)
        else:
            message, articles = self.query_message(query, token_budget=token_budget)
        if self.debug:
            print(message)
        messages = [
            {"role": "system", "content": self.introduction},
            {"role": "user", "content": message},
        ]

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )
        response_message = response.choices[0].message.content

        references = "<p><b>For more information:</b></p><ul>"
        for title, url in articles.items():
            references += "<li><a href=\"" + url  + "\">" + title + "</a></li>"
        references += "</ul>"

        return response_message, references, articles


class ConversationLogger:
    def __init__(
        self,
        # db_path: str = "conversation_log.db",
        db_host = None,
        db_port = None,
        db_user = None,
        db_password = None,
        logs_database_name = None,
        logs_user = None,
        logs_password = None,
        table_name = 'entries', # the name of the table where database entries are logged
        overwrite_db = False,
        debug = False

    ) -> None:
        self.db_host = db_host if db_host is not None else os.getenv('DB_HOST')
        self.db_port = db_port if db_port is not None else os.getenv('DB_PORT')
        self.db_user = db_user if db_user is not None else os.getenv('POSTGRES_USER')
        self.db_password = db_password if db_password is not None else os.getenv('POSTGRES_DB_PASSWORD')
        self.logs_database_name = logs_database_name if logs_database_name is not None else os.getenv('SHELLBOT_DB_NAME')
        self.logs_user = logs_user if logs_user is not None else os.getenv('SHELLBOT_USER')
        self.logs_password = logs_password if logs_password is not None else os.getenv('SHELLBOT_USER_PASSWORD')
        self.table_name = table_name
        self.overwrite_db = overwrite_db
        self.debug = debug
        self.setup_database()

    def database_connection(self):
        conn = psycopg2.connect(
            dbname=self.logs_database_name,
            user=self.logs_user,
            password=self.logs_password,
            host=self.db_host,
            port=self.db_port,
            sslmode='allow'
        )
        cur = conn.cursor()
        return conn, cur

    def create_table(self, table_name, create_table_query):
        conn, cur = self.database_connection()
        cur.execute(create_table_query)
        cur.execute(sql.SQL("GRANT ALL PRIVILEGES ON TABLE {} TO {}").format(
            sql.Identifier(table_name),
            sql.Identifier(self.logs_user)
        ))
        conn.commit()
        cur.close()
        conn.close()

    def setup_database(self):
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            session_id TEXT,
            entry_timestamp TEXT,
            user_input TEXT,
            bot_response TEXT
        );
        """
        self.create_table(self.table_name, create_table_query)

    def post_entry(self, entry):
        conn, cur = self.database_connection()

        # SQL query to insert the data
        insert_query = f"""
        INSERT INTO {self.table_name} (session_id, entry_timestamp, user_input, bot_response)
        VALUES (%s, %s, %s, %s);
        """

        # Execute the query with the data
        cur.execute(insert_query, (
            entry['session_id'],
            entry['timestamp'],
            entry['user_input'],
            entry['bot_response']
        ))

        # Commit the transaction
        conn.commit()

        # Close the cursor and connection
        cur.close()
        conn.close()
        if self.debug:
            print(f"Entry posted successfully to the '{self.table_name}' table.")

    def get_entries(self, limit=0):
        conn, cur = self.database_connection()

        # SQL query to fetch all rows from the 'entries' table
        fetch_query = f"""
        SELECT * FROM {self.table_name};
        """
        if limit > 0:
            fetch_query += " LIMIT " + str(self.limit)
        cur.execute(fetch_query)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Process and print the fetched rows
        results = []
        for row in rows:
            if self.debug:
                print(row)
            results.append({
                'session_id': row[0],
                'entry_timestamp': row[1],
                'user_input': row[2],
                'bot_response': row[3]
            })
        return results


if __name__ == "__main__":
    extractor = WikiExtractor(
        site_name = "www.gem.wiki",
        url = 'https://www.gem.wiki/w/api.php'
    )
    wiki_strings, urls = extractor.compile_wiki_strings()
    openai_client = OpenAI(
        organization='***REMOVED***',
        project='***REMOVED***'
    )

    embedder = Embedder(openai_client)
    df = embedder.compile_embeddings(wiki_strings, urls)

    storage = Storer(openai_client, df, overwrite_db = True, overwrite_pinecone = True)
    query_output = storage.query_article('Clean Coal','content')
    print(query_output)
    content_query_output = storage.query_article("Wipperdorf",'content')
    print(content_query_output)
