--
-- PostgreSQL database dump
--

-- Dumped from database version 14.18 (Homebrew)
-- Dumped by pg_dump version 14.18 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: entries; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entries (
    session_id text,
    entry_timestamp text,
    user_input text,
    bot_response text
);


--
-- Name: gmail_messages; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.gmail_messages (
    id integer NOT NULL,
    subject text,
    "timestamp" integer,
    from_email text,
    to_emails text,
    message text
);


--
-- Name: gmail_messages; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.laptop_documents (
    id integer NOT NULL,
    title text,
    file_path text,
    is_mine boolean,
    last_modified_date text,
    ingested_at integer,
    document_kind text,
    priority_rank int,
    content_hash text

);

--
-- Name: shellbot_knowledge; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.shellbot_knowledge (
    vector_id text NOT NULL,
    platform text,
    title text,
    unix_timestamp integer,
    formatted_datetime text,
    content text,
    url text
);


--
-- Name: social_posts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.social_posts (
    id integer NOT NULL,
    platform text,
    platform_id text,
    "timestamp" text,
    content text,
    url text
);


--
-- Name: gmail_messages gmail_messages_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.gmail_messages
    ADD CONSTRAINT gmail_messages_pkey PRIMARY KEY (id);


--
-- Name: social_posts social_posts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.social_posts
    ADD CONSTRAINT social_posts_pkey PRIMARY KEY (id);


--
-- Name: shellbot_knowledge socialdata_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.shellbot_knowledge
    ADD CONSTRAINT socialdata_pkey PRIMARY KEY (vector_id);


--
-- PostgreSQL database dump complete
--

