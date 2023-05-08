from typing import List
from langchain import PromptTemplate
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
import streamlit as st
from streamlit import components
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, initialize_agent, AgentType
from langchain.chains import ConversationalRetrievalChain, APIChain, summarize, LLMChain
from langchain.chains.api import open_meteo_docs
from langchain.chat_models import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.callbacks import StdOutCallbackHandler, get_openai_callback
from langchain.schema import AgentAction, AgentFinish, LLMResult
from typing import Any, Dict, List, Optional, Union
from langchain.input import print_text

class CustomCallbackHandler(StdOutCallbackHandler):
    def on_new_token(self, token, run_id, parent_run_id):
        print(f"New token: {token}")
    def on_agent_action( self, action: AgentAction, color: Optional[str] = None, **kwargs: Any):
        st.write(action.log)

from sqlalchemy.dialects import registry
registry.load("snowflake")

import os

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

account_identifier = st.secrets["account_identifier"]
user =  st.secrets["user"]
password =  st.secrets["password"]
database_name =  st.secrets["database_name"]
schema_name =  st.secrets["schema_name"]
warehouse_name =  st.secrets["warehouse_name"]
role_name =  st.secrets["role_name"]

conn_string = f"snowflake://{user}:{password}@{account_identifier}/{database_name}/{schema_name}?warehouse={warehouse_name}&role={role_name}"
db = SQLDatabase.from_uri(conn_string)
print("DB===", db)
toolkit = SQLDatabaseToolkit(llm=OpenAI(temperature=0), db=db)

# agent_executor = create_sql_agent(
#     llm=OpenAI(temperature=0),
#     toolkit=toolkit,
#     verbose=True
# )

st.title('Genflow SQL')

sql_agent = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True
)

template_str = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! 
Question: {input}
{agent_scratchpad}
"""

prompt_template = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    template=template_str,
)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
meteo_chain = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=True)

tools = [
    Tool(
        name="SQlAgent",
        func=sql_agent.run,
        description="useful for when you need to answer questions or tasks about querying a database."
    ),
    Tool(
        name="Weather",
         func=meteo_chain.run,
         description="useful for when you need to answer questions about weather, temperature, climate etc"
    )
]

chain = LLMChain(llm=llm, prompt=prompt_template)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

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
    # chain_input = {
    # 'tool_names': 'your_tool_names',
    # 'input': 'your_input',
    # 'tools': '',
    # 'agent_scratchpad': 'your_agent_scratchpad'
    # }

    # chain_result = chain(chain_input)
    #CustomCallbackHandler.on_agent_action(agent.run(query))
    with get_openai_callback() as cb:
        result = agent.run(query)
        st.write(result)
        st.write()
        st.write(cb)
    # result = agent.run(query)
    # print("result -----",result)
