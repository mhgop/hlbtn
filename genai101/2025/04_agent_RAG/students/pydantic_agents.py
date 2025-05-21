# ------------------------------------------------- Provided Code -------------------------------------------------
from pydantic import BaseModel, Field, ValidationError
import asyncio
import os
from openai import OpenAI

async def llm_call(prompt):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=16384,
        temperature=0
    )
    return response.choices[0].message.content

def simple_llm_call(prompt):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=16384,
        temperature=0
    )
    return response.choices[0].message.content

class TextInput(BaseModel):
    theme: str = Field(..., min_length=1, description="The theme of the article.")
    text: str = Field(..., min_length=1, description="The input text for analysis.")

# ------------------------------------------------- Code to be completed -------------------------------------------------
# Create multiple classes that will contain suggestions to help generate the article
# For example, write an agent that will help find examples or arguments.
# Write a last class to correct the syntax, grammar of the article.
# These classes must simply contain a field and the field type

class FirstOutput(BaseModel):
    ""

class SecondOutput(BaseModel):
    ""

class ThirdOutput(BaseModel):
    ""

class ArticleOutput(BaseModel):
    ""

# combine all the above classes here
class FinalResponse(BaseModel):
   ""

class GrammarOutput(BaseModel):
    ""

# Async functions representing agents

async def first_analysis(text: str) -> FirstOutput:
    # Create a prompt and call the llm in a async manner (use await)
    # Extract the information
    return FirstOutput("")

async def second_analysis(theme: str, text: str) -> SecondOutput:
    # Create a prompt and call the llm in a async manner (use await)
    # Extract the information
    return SecondOutput("")

async def third_analysis(theme: str, text: str) -> ThirdOutput:
    # Create a prompt and call the llm in a async manner (use await)
    # Extract the information
    return ThirdOutput("")

async def article_writer(first: FirstOutput, second: SecondOutput, third: ThirdOutput) -> ArticleOutput:
    # Create a prompt combining all the previous outputs and call the llm in a async manner (use await)
    return ArticleOutput("")

# Grammar check function
async def grammar_check(text: str) -> GrammarOutput:
    # Create a prompt and call the llm in a async manner (use await)
    # Extract the information
    return GrammarOutput("")


# Main async function

async def main(input_data: TextInput):
    try:
        # Validate input using Pydantic
        validated_input = TextInput(**input_data.model_dump())

        # Run all agents asynchronously
        first = first_analysis()
        second = second_analysis()
        third = third_analysis()

        # Await intermediate results (use asyncio.gather)
        first_res, second_res, third_res = ""

        # Run article writer agent
        article_task = await article_writer(first_res, second_res, third_res)

        # Optionnal
        # Combine results and validate final response 

        # Grammar correction task
        grammar_task = await grammar_check("")
        print(grammar_task.corrected_text.replace("\\n", "\n"))

    except ValidationError as e:
        print("Validation error:", e.model_dump_json())


if __name__ == "__main__":
    theme = input("Please provide a theme to write an article on: ")
    # Either ask the user to elaborate on the given subject or ask an llm (simple_llm_call) to do so
    description = ""

    # Run the async workflow
    asyncio.run(main(
                    TextInput(
                        theme=theme,
                        text=description
                        )
                    )
                )