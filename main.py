# from typing import List
# from langchain import PromptTemplate
# from langchain.agents import create_sql_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.sql_database import SQLDatabase
# from langchain.llms.openai import OpenAI
# from langchain.agents import AgentExecutor
# import streamlit as st
# from streamlit import components
# from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, initialize_agent, AgentType
# from langchain.chains import ConversationalRetrievalChain, APIChain, summarize, LLMChain
# from langchain.chains.api import open_meteo_docs
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import StringPromptTemplate
# from langchain.callbacks import StdOutCallbackHandler, get_openai_callback
# from langchain.schema import AgentAction, AgentFinish, LLMResult
# from typing import Any, Dict, List, Optional, Union
# from langchain.input import print_text

# class CustomCallbackHandler(StdOutCallbackHandler):
#     def on_new_token(self, token, run_id, parent_run_id):
#         print(f"New token: {token}")
#     def on_agent_action( self, action: AgentAction, color: Optional[str] = None, **kwargs: Any):
#         st.write(action.log)

# handler = StdOutCallbackHandler()

# from sqlalchemy.dialects import registry
# registry.load("snowflake")

# import os

# os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

# account_identifier = st.secrets["account_identifier"]
# user =  st.secrets["user"]
# password =  st.secrets["password"]
# database_name =  st.secrets["database_name"]
# schema_name =  st.secrets["schema_name"]
# warehouse_name =  st.secrets["warehouse_name"]
# role_name =  st.secrets["role_name"]

# conn_string = f"snowflake://{user}:{password}@{account_identifier}/{database_name}/{schema_name}?warehouse={warehouse_name}&role={role_name}"
# db = SQLDatabase.from_uri(conn_string)
# print("DB===", db)
# toolkit = SQLDatabaseToolkit(llm=OpenAI(temperature=0), db=db)

# # agent_executor = create_sql_agent(
# #     llm=OpenAI(temperature=0),
# #     toolkit=toolkit,
# #     verbose=True
# # )

# st.title('Genflow SQL')

# sql_agent = create_sql_agent(
#     llm=OpenAI(temperature=0),
#     toolkit=toolkit,
#     verbose=True,
#     max_iterations=5
# )

# template_str = """
# Answer the following questions as best you can. You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin! 
# Question: {input}
# {agent_scratchpad}
# """

# prompt_template = PromptTemplate(
#     input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
#     template=template_str,
# )
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# meteo_chain = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=True)

# tools = [
#     Tool(
#         name="SQlAgent",
#         func=sql_agent.run,
#         description="useful for when you need to answer questions or tasks about querying a database."
#     ),
#     Tool(
#         name="Weather",
#          func=meteo_chain.run,
#          description="useful for when you need to answer questions about weather, temperature, climate etc"
#     )
# ]

# chain = LLMChain(llm=llm, prompt=prompt_template)
# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# # Create a sidebar with a dropdown menu
# selected_table = st.sidebar.selectbox(
#     "Select a table:", options=list(db._usable_tables))
# st.sidebar.markdown(f"### DDL for {selected_table} table")
# st.sidebar.code(db.get_table_info([selected_table]), language="sql")

# with st.form(key='my_form'):
#     query = st.text_input("Query: ", key="input", value="",
#                           placeholder="Type your query here...", label_visibility="hidden")
#     submit_button = st.form_submit_button(label='Submit')
# col1, col2 = st.columns([1, 3.2])
# reset_button = col1.button("Reset Chat History")

# if len(query) > 2 and submit_button:
#     # chain_input = {
#     # 'tool_names': 'your_tool_names',
#     # 'input': 'your_input',
#     # 'tools': '',
#     # 'agent_scratchpad': 'your_agent_scratchpad'
#     # }

#     # chain_result = chain(chain_input)
#     #CustomCallbackHandler.on_agent_action(agent.run(query))
#     result = agent.run(query, callbacks=[handler])
#     # with get_openai_callback() as cb:
#     #     result = agent.run(query)
#     #     st.write(result)
#     #     st.write()
#     #     st.write(cb)
#     # result = agent.run(query)
#     print("result -----",result)

import os
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.chains import SQLDatabaseSequentialChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.prompts.prompt import PromptTemplate
import streamlit as st
import pandas as pd
from streamlit import components
import json
from sqlalchemy.dialects import registry
registry.load("snowflake")

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

account_identifier = st.secrets["account_identifier"]
user =  st.secrets["user"]
password =  st.secrets["password"]
database_name =  st.secrets["database_name"]
schema_name =  st.secrets["schema_name"]
warehouse_name =  st.secrets["warehouse_name"]
role_name =  st.secrets["role_name"]

conn_string = f"snowflake://{user}:{password}@{account_identifier}/{database_name}/{schema_name}?warehouse={warehouse_name}&role={role_name}"

db = SQLDatabase.from_uri(conn_string, sample_rows_in_table_info=0)
decider_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, verbose=True)
db_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, verbose=True)

#decider_llm = OpenAI(temperature=0.7, verbose=True)
#db_llm = OpenAI(temperature=0.3, verbose=True)
#llm = OpenAI(temperature=0.7, verbose=True)

# chain = SQLDatabaseSequentialChain.from_llm(llm, db, verbose=True)
# result = chain.run("Identify why customers switched from using one software to another.")
# print("RESULT : ", result)

st.title('Genflow Snoflake')

def dbExec(user_query:str):
    #user_query = "Get product reviews along with product name and vendor name"

    DECIDER_TEMPLATE = """Given the below input question and list of potential tables and their description, output a comma separated list of the table names that may be necessary to answer this question.

    Question: {query}

    Table Names: {table_names}

    Relevant Table Names:"""

    DECIDER_PROMPT = PromptTemplate(
        input_variables=["query", "table_names"],
        template=DECIDER_TEMPLATE,
        output_parser=CommaSeparatedListOutputParser(),
    )
    decider_chain = LLMChain(
                llm=decider_llm, prompt=DECIDER_PROMPT, output_key="table_names"
            )
    llm_inputs = {
                "query": user_query,
                "table_names": '''{
        "product_reviews_benchmark" : "This table contains data for product reviews and various associated metrics, grouped by product and quarter. Key attributes include product ID, quarter, number of reviews, whether it meets requirements, its price, and multiple metrics for ease of setup, administration, and use. It also tracks the quality of support, net promoter score (NPS), and segmentation data. Additionally, it contains information on frequency of use, the timeframe to go live, and different parameters related to the number of users and costs involved.",

        "category_product" : "This table is a bridge table between products and their categories. It records the product ID, the associated category ID, as well as the category name and type. This table is useful in retrieving information on the category or categories that a specific product falls under.",

        "category" : "This table provides data for the categories themselves. Important columns include the category's unique ID, name, creation and update timestamps, and whether it's a featured category. It also includes metadata like the category's SEO type, slug, keywords, and more. The table additionally records the count of reviews and products associated with each category.",

        "vendor_reviews_benchmark" : "This table parallels the structure of the PRODUCT_REVIEWS_BENCHMARK table, but reviews are associated with vendors instead of individual products. Key columns include vendor ID, quarter, review count, star rating, whether it meets requirements, its price, and various metrics for ease of setup, administration, and use. It also captures the quality of support, net promoter score (NPS), segmentation data, and other parameters such as frequency of use, go-live timelines, user counts, and costs.",

        "product" : "This table details individual product data. Significant columns include the product ID, type, name, URL, description, count of survey responses, creation and update timestamps. It also stores information about the product's vendor, including the vendor's name, URL, and ID. The main category ID and array of all associated categories are also included.",

        "review": "This table is dedicated to individual reviews for products. Key attributes are the survey response ID, product ID, submission and update timestamps, star rating, categories, company segment, and a variety of scores for aspects like meets requirements, price, ease of setup, etc. It also contains the reviewers' comments, their roles, and details about their company size and industry. Use this table for questions related to star ratings",

        "vendor" : "This table contains vendor-specific data. Notable columns include vendor ID, name, LinkedIn company page URL, headquarters location, founding year, company website, LinkedIn followers, Twitter handle, and Glassdoor rating. There is also information about the number of reviews for the vendor, the number of their products, and company-level metadata like ownership, annual revenue, and year of revenue data."
    }''',
            }

    result = decider_chain.predict_and_parse(**llm_inputs)
    print("SHORTLISTED TABLES : ", result)

    PROMPT_SUFFIX = """Only use the following tables:
    {table_info}

    Question: {input}"""

    QUERY_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

    Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

    Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.


    Use the following format:

    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here (Give the answer as JSON, if more than one element then give as Array of JSON objects. Look at the SQLQuery that you generated and get the column names from there, make them as keys in JSON object). Return the complete answer, don't return incomplete answers, don't use ...

    """

    QUERY_PROMPT = PromptTemplate(
        input_variables=["input", "table_info", "dialect", "top_k"],
        template=QUERY_TEMPLATE + PROMPT_SUFFIX,
    )

    new_inputs = {
                "query": user_query,
                "table_names_to_use": result,
            }
    sql_chain = SQLDatabaseChain.from_llm(
                llm=db_llm, db=db, prompt=QUERY_PROMPT, return_direct=False, top_k=10, verbose=True
            )
    try:
        final_result = sql_chain.run(
                **new_inputs, return_intermediate_steps=True, use_query_checker=True, return_only_outputs=True
            )
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
        res = dbExec(query)
        json_data = json.loads(res)
        # Proceed with using the JSON data
        st.json(json_data)
        df = pd.DataFrame(json_data)
        st.table(df)
    except Exception as e:
        # Handle the JSON decoding error
        #st.code(res, language="json", line_numbers=False)
        st.error(e, icon="🚨")