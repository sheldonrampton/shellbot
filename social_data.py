from openai import OpenAI # for calling the OpenAI API
import sqlite3
import datetime
import re
from bs4 import BeautifulSoup
import os
import pandas as pd
import tiktoken  # for counting tokens
from itertools import islice
from pinecone import Pinecone, ServerlessSpec
from chatbotter import BatchGenerator, Asker
import warnings
import hashlib
import time

# Suppress specific DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


sig_lines = ["""
--
Sheldon Rampton
"I think, therefore I procrastinate."
(608) 206-2745
"""]

sanitizations = [
    "stephanierae1106",
    "CanyonTherapy@gmail.com",
    "secretagentsidekick@gmail.com",
    "canyontherapy@gmail.com",
    "Backpage",
    "secretagentsidekick.com",
    "Korinne Kaiser",
    "Shameless80",
    "TommyMke",
    "subvertoursenses@gmail.com",
    "vivianhetaira",
    "adultlook",
    "erotic review",
    "mydreamydanielle@gmail.com",
    "BookTheDT@yahoo.com",
    "juliettestouches@gmail.com",
    "laura.lux1000@gmail.com",
    "adultfriendfinder"
]

class SocialData:
    def __init__(self,
        openai_client,
        batch_size = 1000,
        embedding_model = "text-embedding-3-small",
        social_db_path: str = "social_media.db",
        gmail_db_path: str = 'gmail.db',
        excluded_text = [],
        sanitizations = [],
        gpt_model: str = "gpt-4o",  # selects which tokenizer to use
        pinecone_index_name = "shellbot-embeddings2",
        overwrite_pinecone = False,
        db_path = 'shellbot2.db',
        overwrite_db = False,
        limit = 0,
        debug = False
    ) -> None:
        self.openai_client = openai_client
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.social_db_path = social_db_path
        self.gmail_db_path = gmail_db_path
        self.excluded_text = excluded_text
        self.sanitizations = sanitizations
        self.gpt_model = gpt_model
        self.pinecone_index_name = pinecone_index_name
        self.overwrite_pinecone = overwrite_pinecone
        self.db_path = db_path
        self.overwrite_db = overwrite_db
        self.debug = debug
        self.limit = limit
  
    def fetch_data(self):
        """
        Retrieves the contents of the social_media and gmail databases
        and combines them into a single 
        """

        # Connect to the SQLite databases
        conn_social = sqlite3.connect(self.social_db_path)
        conn_gmail = sqlite3.connect(self.gmail_db_path)
        
        # Create cursors for each connection
        cursor_social = conn_social.cursor()
        cursor_gmail = conn_gmail.cursor()
        
        # SQL query for SocialPosts table
        query_social = """
        SELECT 'social' AS source, id, platform, platform_id as title,
        CAST(timestamp AS INTEGER) AS unix_timestamp, 
        content, url
        FROM SocialPosts 
        """
        if self.limit > 0:
            query_social += " LIMIT " + str(self.limit)
        # Execute the queries
        social_results = cursor_social.execute(query_social).fetchall()
        # Load the results into a Pandas DataFrame
        df_social = pd.DataFrame(social_results, columns=[
            'source', 'id', 'platform', 'title', 'unix_timestamp',
            'content', 'url'
        ])
        df_social['content'] = df_social.apply(lambda row: row['url'] if row['content'] == '' else row['content'], axis=1)
        df_social['title'] = df_social['content'].apply(lambda x: x.split('\n')[0])
        df_social['title'] = df_social['title'].apply(lambda x: self.truncated_string(x, max_tokens=30))

        # SQL query for GmailMessages table
        query_gmail = """
        SELECT 'gmail' AS source, id, 'Email' as platform, subject as title,
        timestamp AS unix_timestamp,
        message as content, from_email, to_emails 
        FROM GmailMessages 
        """
        if self.limit > 0:
            query_gmail += " LIMIT " + str(self.limit)
        gmail_results = cursor_gmail.execute(query_gmail).fetchall()
        # Load the results into a Pandas DataFrame
        df_gmail = pd.DataFrame(gmail_results, columns=[
            'source', 'id', 'platform', 'title', 'unix_timestamp',
            'content', 'from_email', 'to_emails'
        ])
        df_gmail = df_gmail[~df_gmail['content'].isin(self.sanitizations)]
        pattern = '|'.join(self.sanitizations)
        # Remove rows where the "content" column contains any of the strings in the list
        df_gmail = df_gmail[~df_gmail['content'].str.contains(pattern, na=False)]

        # Combine 'from_email' and 'to_emails' into a single column named 'participants'
        df_gmail['url'] = df_gmail['from_email'] + ', ' + df_gmail['to_emails']
        # Optionally, drop the original 'from_email' and 'to_emails' columns if they are no longer needed
        df_gmail = df_gmail.drop(columns=['from_email', 'to_emails'])

        # # Now df has the combined 'participants' column
        df_combined = pd.concat([df_gmail, df_social], ignore_index=True)

        df_combined['datetime'] = df_combined['unix_timestamp'].apply(self.format_timestamp)
        df_combined['title'] = df_combined['datetime'] + " " + df_combined['platform'] + ": " + df_combined['title']

        # Close the database connections
        conn_social.close()
        conn_gmail.close()
        
        df = df_combined.sort_values(by='unix_timestamp', ascending=True)
        # Remove duplicate content
        df = df.drop_duplicates(subset='content', keep='first')

        # Apply clean_message() to 'content' column where 'platform' is "Email"
        df.loc[df['platform'] == 'Email', 'content'] = df.loc[df['platform'] == 'Email', 'content'].apply(self.cleanup_email)
        df['content'] = df['content'].apply(lambda x: self.truncated_string(x))
        df['url'] = df['url'].fillna('')
        df['title'] = df['title'].fillna('No title')
        df["embedding"] = df.apply(lambda row: self.openai_client.embeddings.create(model=self.embedding_model, input=row['content']).data[0].embedding, axis=1)
        df["vector_id"] = df.apply(lambda row: self.generate_vector_id(str(row['title']) + str(row['content'])), axis=1)
        self.df = df.drop(columns=['source', 'id'])
        return df

    def is_html(self, content):
        """
        Returns true of the content is HTML, false otherwise.
        """

        # Pattern to detect minimal HTML tags
        html_pattern = re.compile(r'<\s*([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>(.*?)<\s*/\1\s*>', re.IGNORECASE | re.DOTALL)
        # Search for the pattern in the content
        return bool(html_pattern.search(content))

    def strip_html(self, html_content):
        """
        Converts HTML to plain text.
        """

        # Use BeautifulSoup to parse the HTML content
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove custom tags and their contents if needed
        for tag in soup.find_all(['x-stuff-for-pete', 'x-sigsep']):
            tag.decompose()
        
        # Extract text from the parsed HTML
        text = soup.get_text(separator='\n', strip=True)
        return text

    def clean_message(self, text):
        """
        Cleans some unwanted text out of email message bodies,
        including sig lines and > characters at the beginning of
        quoted text.
        """

        # Split the message into lines
        lines = text.split('\n')
        
        # Process each line to handle quotes and clean text
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Check if the line starts with '>' and handle accordingly
            if line.startswith('>'):
                # You can uncomment the next line if you want to skip quoted lines
                # continue
                # Or you can remove '>' to clean it but keep the text
                line = line.lstrip('> ').lstrip('> ')  # Remove leading '>' and extra spaces
            elif line.startswith('[') and 'image:' in line:
                # Skip lines that are just image markers or similar non-text content
                continue
            if line.startswith('<') and line.endswith('>'):
                line = line[1:-1]
            cleaned_lines.append(line)

        # Join the cleaned lines back into a single string
        cleaned_text = '\n'.join(cleaned_lines)
        for text_to_remove in self.excluded_text:
            cleaned_text = cleaned_text.replace(text_to_remove, '')
        cleaned_text = cleaned_text.replace("\n\n\n\n\n", "\n\n").replace("\n\n\n\n", "\n\n")
        cleaned_text = cleaned_text.replace("\n\n\n", "\n\n")
        return cleaned_text

    def cleanup_email(self, content):
        if self.is_html(content):
            return self.strip_html(content)
        else:
            return self.clean_message(content)

    def format_timestamp(self, unix_timestamp):
        date_time = datetime.datetime.fromtimestamp(int(unix_timestamp))
        # Format the datetime object as "Month Day, Year"
        return date_time.strftime("%B %d, %Y")

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
        max_tokens: int = 7500
    ) -> str:
        """Truncate a string to a maximum number of tokens."""
        encoding = tiktoken.encoding_for_model(self.gpt_model)
        encoded_string = encoding.encode(string)
        truncated_string = encoding.decode(encoded_string[:max_tokens])
        if self.debug and len(encoded_string) > max_tokens:
            first_line = string.split('\n')[0]
            print(f"Warning: Truncated string {first_line} from {len(encoded_string)} tokens to {max_tokens} tokens.")
        return truncated_string

    def split_strings_from_message(
        self,
        string: str,
        max_tokens: int = 7500,
        max_recursion: int = 5,
    ) -> list[str]:
        """
        Split a subsection into a list of subsections, each with no more than max_tokens.
        Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
        """
        num_tokens_in_string = self.num_tokens(string)
        # if length is fine, return string
        if num_tokens_in_string <= max_tokens:
            return [string]
        # if recursion hasn't found a split after X iterations, just truncate
        elif max_recursion == 0:
            return [self.truncated_string(string, max_tokens=max_tokens)]
        # otherwise, split in half and recurse
        else:
            for delimiter in ["\n\n", "\n", ". "]:
                left, right = self.halved_by_delimiter(string, delimiter=delimiter)
                if left == "" or right == "":
                    # if either half is empty, retry with a more fine-grained delimiter
                    continue
                else:
                    # recurse on each half
                    results = []
                    for half in [left, right]:
                        half_subsection = half
                        half_strings = self.split_strings_from_message(
                            half_subsection,
                            max_tokens=max_tokens,
                            max_recursion=max_recursion - 1,
                        )
                        results.extend(half_strings)
                    return results
        # otherwise no split was found, so just truncate (should be very rare)
        return [self.truncated_string(string, max_tokens=max_tokens)]

    def generate_vector_id(self, text: str) -> str:
        # Create a SHA256 hash object
        hash_object = hashlib.sha256()
        
        # Update the hash object with the text encoded in UTF-8
        hash_object.update(text.encode('utf-8'))
        
        # Return the hexadecimal digest of the hash, which is a string representation of the hash
        return hash_object.hexdigest()

    def compile_embeddings(self, strings, df):
        embeddings = []
        df["embedding"] = df.apply(lambda row: self.openai_client.embeddings.create(row['text']), axis=1)
        df["vector_id"] = df.apply(lambda row: self.generate_vector_id(row['title'] + row['text']), axis=1)

        if self.debug:
            for value in df['title']:
                print(value)
            for value in df['url']:
                print(value)
            for value in df['vector_id']:
                print(value)

        return df

    def setup_database(self):
        if os.path.exists(self.db_path) and self.overwrite_db:
            os.remove(self.db_path)
        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
            CREATE TABLE IF NOT EXISTS SocialData (
                vector_id TEXT PRIMARY KEY,
                platform TEXT,
                title TEXT,
                unix_timestamp INT,
                formatted_datetime TEXT,
                content TEXT,
                url TEXT
            )
            ''')
            conn.commit()
            conn.close()

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
        if self.debug:
            print("Uploading vectors to content namespace..")
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
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
                try:
                    c.execute('''
                    INSERT INTO SocialData (vector_id, platform, title, unix_timestamp, 
                    formatted_datetime, content, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (row['vector_id'], row['platform'], row['title'], row['unix_timestamp'],
                    row['datetime'], row['content'], row['url']))
                    if self.debug:
                        print("Inserted row ", rownum, row['vector_id'], row['title'])
                except sqlite3.Error as e:
                    print(f"An error occurred: {e}")
                    print(row)
        conn.commit()
        conn.close()
        if self.debug:
            print("Records inserted successfully.")
            # Check index size for each namespace to confirm all of our docs have loaded
            print(self.pinecone_index.describe_index_stats())

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
        df['title'], df['url'], df['content'] = zip(*df['id'].apply(lambda x: self.get_item(x)))
        
        if self.debug:
            counter = 0
            for k,v in df.iterrows():
                counter += 1
                print(f'{v.title} (score = {v.score})')
            
            print('\n')

        return df

    def get_item(self, unique_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # SQL query to retrieve the row with the specified unique_id
        query = "SELECT title, url, content FROM SocialData WHERE vector_id = ?"
        try:
            # Execute the query and fetch the row
            cursor.execute(query, (unique_id, ))
            row = cursor.fetchone()
            conn.close()

            # Check if a row was found
            if row:
                return row
            else:
                print(f"No row found with unique_id = {unique_id}")
                return ['', '', '']
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return None

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

        # if self.debug:
        # print(f'\nMost similar results to {query} in "content" namespace:\n')
        # if not query_result.matches:
        #     print('no query result')
        
        matches = query_result.matches
        ids = [res.id for res in matches]
        scores = [res.score for res in matches]
        df = pd.DataFrame({'id':ids, 
                           'score':scores,
                           })
        
        df['title'], df['url'], df['content'] = zip(*df['id'].apply(lambda x: self.get_item(x)))
        return df


"""
Main loop.
"""
if __name__ == "__main__":
    # Create the "months" directory if it doesn't exist
    openai_client = OpenAI(
      organization='***REMOVED***',
      project='***REMOVED***',
    )
    sd = SocialData(openai_client, excluded_text = sig_lines, sanitizations = sanitizations, overwrite_pinecone = True, overwrite_db = True)
    df = sd.fetch_data()
    # print(sd.df.head(50))
    # nan_rows = df[df['url'].isna()]
    # nan_rows = df[df['title'].isna()]
    # nan_rows = df[df['content'].isna()]
    # print(nan_rows['title'])
    # print(nan_rows['content'])
    # print(nan_rows['datetime'])
    # print(df[df['url'] == "Sheldon Rampton <sheldonmrampton@gmail.com>, Jaye Straus <jaye@strauslaw.com>"].head(100))
    # print(df[df['platform'] == 'Facebook post'].head(100))
    # print(df[df['platform'] == 'Facebook comment'].head(100))
    # print(df[df['platform'] == 'Tweet'].head(100))
    # for i, row in df.iterrows():
    #     print(row['title'])
    #     print(row['embedding'])
        # for piece in pieces:
        #     print(piece)

    sd.setup_database()
    sd.setup_pinecone()
    # sd.debug = False
    sd.upsert_data()
    print(sd.query_article('Portage tennis','content'))
    print(sd.query_article('Expert in Artificial Intelligence','content'))

    asker = Asker(openai_client, storage = sd,
        introduction = 'Use the below messages which were written by Sheldon Rampton to answer questions as though you are Sheldon Rampton. If the answer cannot be found in the articles, write "I could not find an answer."',
        string_divider = 'Messages:'
    )
    response, references, articles = asker.ask("Tell me about Portage tennis.")
    print(response)

    response, references, articles = asker.ask("What have you been doing with regard to artificial intelligence?")
    print(response)
