from langchain.sql_database import SQLDatabase
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
import urllib.parse
import os
import json
import pandas as pd
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.chains.llm import LLMChain
import streamlit as st
from sqlalchemy.dialects import registry

user = st.secrets["mysql_user"]
password = st.secrets["mysql_password"]
host = st.secrets["mysql_host"]
port = st.secrets["mysql_port"]
dbname = st.secrets["mysql_dbname"]

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

encoded_password = urllib.parse.quote_plus(password)

connection_string = f"mysql+mysqlconnector://{user}:{encoded_password}@{host}:{port}/{dbname}"
db = SQLDatabase.from_uri(connection_string, sample_rows_in_table_info=1)

llm =  ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, verbose=True)

st.title('Genflow MYSQL')

def get_salesforce_date(query:str):

    DECIDER_TEMPLATE = """Given the below input question and list of potential tables, output a comma separated list of the table names that may be necessary to answer this question.

    Question: {query}

    Table Names: {table_names}

    Relevant Table Names:"""

    DECIDER_PROMPT = PromptTemplate(
        input_variables=["query", "table_names"],
        template=DECIDER_TEMPLATE,
        output_parser=CommaSeparatedListOutputParser(),
    )


    decider_chain = LLMChain(
                llm=llm, prompt=DECIDER_PROMPT, output_key="table_names"
            )


    llm_inputs = {
                "query": query,
                "table_names": '''{
                    "opportunities" : "This table contains details about opportunities, example the opportunity id, account id, closeDate, amount",
                    "accounts" : "This table contains details about account, example the account id, account name",
                    "tickets" : "This table contains details about tickets created by contact, example the  ticketid, contact id, subject, priorirty",
                    "contacts" : "This table conatins details about contacts created for every account, example contactID, accountID, firstName, lastName",
                    "users" : "This table contains details about the salesforce owner, example id, userName, email
                }''',
            }

    result = decider_chain.predict_and_parse(**llm_inputs)
    print("Result: Table names to be used", result)


    PROMPT_SUFFIX = """Only use the following tables:
    {table_info}

    Question: {input}"""

    # QUERY_TEMPLATE = """You are a SQL query generator which generates the query based on the table provided in the prompt and the actual prompt. SQL query shouldn't have any unwanted conditions or fields and if you dont know the answer, answer should be "DO NOT KNOW".


    QUERY_TEMPLATE = """ You are a SQL query generator which generates the query based on the table provided in the prompt and the actual prompt. SQL query shouldn't have any unwanted conditions or fields and if you dont know the answer, answer should be "DO NOT KNOW".

    Adding an example for your reference within a tripple quotes

    '''
    Prompt: Find the average opportunity value for accounts that have more than 2 contacts in the table 'contact'. Here is the schema of the relevant tables -

    Table 1 - accounts:

    Description: This table contains the list of Salesforce accounts belonging to the customer.
    Schema:
        accountID: a unique identifier and primary key for Salesforce accounts. It is stored as a character string (varchar) and must have a non-null value. 
        accountName: represents the name of the Salesforce account and is stored as a string.
        industry: The industry associated with the Salesforce account (stored as string).
        phone: The primary phone number for the Salesforce account (stored as string).
        billingCity: The billing city for the Salesforce account, which could be the primary office location (stored as string).
        billingState: The billing state for the Salesforce account, which could be the primary office location (stored as string).
        domain: The main website of the Salesforce account. The value can be a URL, and it is stored as a string. Parsing the domain is possible if needed.


    Table 2 - opportunities:

    Description: This table contains the list of Salesforce opportunities associated with each Salesforce account.
    Schema:
        opportunityID: Unique identifier and primary key for Salesforce opportunities (stored as string). It must have a non-null value and is designated as the primary key.
        accountID: The Salesforce account ID to which the opportunity belongs (stored as string). It must have a non-null value and is a foreign key referencing the accountID column in the accounts table.
        name: The name of the Salesforce opportunity (stored as string).
        amount: The value of the opportunity (stored as decimal).
        closeDate: The closing date of the Salesforce opportunity (stored as date).
        stageName: The current stage of the Salesforce opportunity (stored as string).
        userID: The Salesforce account ID to whom the opportunity is assigned (stored as string). It must have a non-null value and is a foreign key referencing the accountID column in the accounts table.

    Table 3 - contacts:

    Description: This table contains the list of Salesforce contacts associated with a Salesforce account.
    Schema:
        contactID: Unique identifier and primary key for Salesforce contacts (stored as string). It must have a non-null value and is designated as the primary key.
        accountID: The Salesforce account ID to which the contact belongs (stored as string). It must have a non-null value and is a foreign key referencing the accountID column in the accounts table.
        email: The primary email address of the Salesforce contact (stored as string). It must have a non-null value.
        firstName: The first name of the Salesforce contact (stored as string).
        lastName: The last name of the Salesforce contact (stored as string).
        phone: The primary phone number of the Salesforce contact (stored as string).

    Table 4 - tickets:

    Description: This table contains the list of Salesforce tickets.
    Schema:
        ticketID: The unique identifier for each ticket (stored as VARCHAR) and designated as the primary key.
        contactID: The identifier of the contact associated with the ticket (stored as VARCHAR) and referenced as a foreign key to the contactID column in the contacts table.
        status: The status of the ticket, such as "Open," "In Progress," or "Closed" (stored as VARCHAR).
        subject: The subject of the ticket (stored as VARCHAR).
        description: A detailed description or content of the ticket (stored as TEXT).
        priority: The priority (level) of the ticket, such as "Low," "Medium," or "High" (stored as VARCHAR).

    Table 5 - users:

    Description: This table contains the list of Salesforce users.
    Schema:
        id: The unique identifier for each user (stored as VARCHAR) and designated as the primary key.
        username: The username of the user (stored as VARCHAR).
        email: The email of the user or owner (stored as VARCHAR).
        full_name: The full name of the user (stored as VARCHAR).
        user_role: The role of the user (stored as VARCHAR).
        profile: The profile information of the user (stored as VARCHAR).
        created_date: The date when the user was created (stored as DATE).
        last_login_date: The date of the user's last login (stored as DATE).
        is_active: Indicates whether the user is active or not (stored as BOOLEAN).
        time_zone: The time zone of the user (stored as VARCHAR).

        '''

    
    Use the following format:

    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here (Give the answer as JSON, if more than one element then give as Array of JSON objects. Look at the SQLQuery that you generated and get the column names from there, make them as keys in JSON object). Return the complete answer, don't return incomplete answers, don't use ...
    """


    QUERY_PROMPT = PromptTemplate(
        input_variables=["input", "table_info"],
        template=QUERY_TEMPLATE + PROMPT_SUFFIX
    )

    new_inputs = {
                "query": query,
                "table_names_to_use": result,
            }

    print("*****", new_inputs)

    sql_chain = SQLDatabaseChain.from_llm(
                llm=llm, db=db, prompt=QUERY_PROMPT, verbose=True
            )
    try:
        final_result = sql_chain.run(
                **new_inputs, return_intermediate_steps=True, use_query_checker=True,return_only_outputs=True)
        return final_result
    except Exception as e:
        #return e
        raise ValueError(str(e))

# Create a sidebar with a dropdown menu
selected_table = st.sidebar.selectbox(
    "Select a table:", options=list(db._usable_tables))
st.sidebar.markdown(f"### DDL for {selected_table} table")
st.sidebar.code(db.get_table_info([selected_table]), language="sql")

with st.form(key='my_form'):
    query = st.text_input("Query: ", key="input", value="",
                          placeholder="Type your query here...", label_visibility="hidden")
    submit_button = st.form_submit_button(label='Submit')
col1, col2 = st.columns([1, 3.2])
reset_button = col1.button("Reset Chat History")

if len(query) > 2 and submit_button:
    try:
        res = get_salesforce_date(query)
        json_data = json.loads(res)
        # Proceed with using the JSON data
        st.json(json_data)
        df = pd.DataFrame(json_data)
        st.table(df)
    except Exception as e:
        # Handle the JSON decoding error
        #st.code(res, language="json", line_numbers=False)
        st.error(e, icon="🚨")