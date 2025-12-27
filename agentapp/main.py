from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from tools import save_to_txt, wiki_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm_google = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0)
llm_openai = ChatOpenAI(model="gpt-4o", temperature=0)  
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

tools = [wiki_tool, save_to_txt]

agent = create_agent(
    model=llm_google,
    tools=tools,
    system_prompt=f"""
        You are a research assistant that will help generate a research paper.
        Answer the user query and use neccessary tools. 
        Wrap the output in this format and provide no other text\n{parser.get_format_instructions()}
    """,  
    response_format=parser.pydantic_object  
)
user_query = input("What can I help you research? ")
messages = [{"role": "user", "content": user_query}]
raw_response = agent.invoke({"messages": messages})



try:
    structured_response = raw_response.get("structured_response")
    print("Topic:", structured_response.topic)
    print("Summary:", structured_response.summary)
    print("Sources:", structured_response.sources)
    print("Tools used:", structured_response.tools_used)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)