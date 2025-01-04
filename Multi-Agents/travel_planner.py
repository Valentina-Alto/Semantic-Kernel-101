import asyncio

from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
from semantic_kernel.kernel import Kernel

from services import Service

from service_settings import ServiceSettings

service_settings = ServiceSettings.create()

# Select a service to use for this notebook (available services: OpenAI, AzureOpenAI, HuggingFace)
selectedService = (
    Service.AzureOpenAI
    if service_settings.global_llm_service is None
    else Service(service_settings.global_llm_service.lower())
)

###################################################################


CONCIERGE_NAME = "Concierge"
CONCIERGE_INSTRUCTIONS = """
You are a concierge for a travel agency specializing in hiking trips in Kyrgyzstan.
Your goal is to assist users in planning their hiking trips by coordinating with experts in route planning and local traditions.
Your role is to facilitate the conversation and ensure all participants contribute to the travel plan.
Whenever the user greets or thank you, feel free to answer without invoking other agents.
"""

KYRGYZSTAN_ROUTE_EXPERT_NAME = "KyrgyzstanRouteExpert"
KYRGYZSTAN_ROUTE_EXPERT_INSTRUCTIONS = """
You are an expert in hiking routes in Kyrgyzstan.
Your goal is to provide detailed information on the best hiking routes in Kyrgyzstan.
Highlight the difficulty levels, scenic spots, and any important landmarks.
"""

LOCAL_TRADITIONS_EXPERT_NAME = "LocalTraditionsExpert"
LOCAL_TRADITIONS_EXPERT_INSTRUCTIONS = """
You are an expert in the local traditions and culture of Kyrgyzstan.
Your goal is to provide insights on local customs and traditions that hikers should be aware of.
Emphasize respectful behavior, cultural norms, and any important local practices.
"""

def _create_kernel_with_chat_completion(service_id: str) -> Kernel:
    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(service_id=service_id))
    
    
    return kernel

async def main():
    while True:
        agent_concierge = ChatCompletionAgent(
            service_id="concierge",
            kernel=_create_kernel_with_chat_completion("concierge"),
            name=CONCIERGE_NAME,
            instructions=CONCIERGE_INSTRUCTIONS,
        )

        agent_kyrgyzstan_route_expert = ChatCompletionAgent(
            service_id="kyrgyzstan_route_expert",
            kernel=_create_kernel_with_chat_completion("kyrgyzstan_route_expert"),
            name=KYRGYZSTAN_ROUTE_EXPERT_NAME,
            instructions=KYRGYZSTAN_ROUTE_EXPERT_INSTRUCTIONS,
        )

        agent_local_traditions_expert = ChatCompletionAgent(
            service_id="local_traditions_expert",
            kernel=_create_kernel_with_chat_completion("local_traditions_expert"),
            name=LOCAL_TRADITIONS_EXPERT_NAME,
            instructions=LOCAL_TRADITIONS_EXPERT_INSTRUCTIONS,
        )

        termination_function = KernelFunctionFromPrompt(
            function_name="termination",
            prompt="""
Determine if the travel plan has been agreed upon by all participants. If so, respond with a single word: yes

History:
{{$history}}
""",
        )

        selection_function = KernelFunctionFromPrompt(
            function_name="selection",
            prompt=f"""
Determine which participant takes the next turn in a conversation based on the most recent participant.
State only the name of the participant to take the next turn.
No participant should take more than one turn in a row.

Always follow these rules when selecting the next participant:
- After user input, it is {CONCIERGE_NAME}'s turn.
- {CONCIERGE_NAME} will then decide whether to invoke {KYRGYZSTAN_ROUTE_EXPERT_NAME} and {LOCAL_TRADITIONS_EXPERT_NAME}.
- Participants take turns sharing their perspectives.
- Ensure all participants have an opportunity to contribute.
- The conversation cycles through participants if necessary.

History:
{{{{$history}}}}
""",
        )

        chat = AgentGroupChat(
            agents=[
                agent_concierge,
                agent_kyrgyzstan_route_expert,
                agent_local_traditions_expert,
            ],
            termination_strategy=KernelFunctionTerminationStrategy(
                agents=[
                    agent_concierge,
                    agent_kyrgyzstan_route_expert,
                    agent_local_traditions_expert,
                ],
                function=termination_function,
                kernel=_create_kernel_with_chat_completion("termination"),
                result_parser=lambda result: str(result.value[0]).lower() == "yes",
                history_variable_name="history",
                maximum_iterations=20,
            ),
            selection_strategy=KernelFunctionSelectionStrategy(
                function=selection_function,
                kernel=_create_kernel_with_chat_completion("selection"),
                result_parser=lambda result: str(result.value[0]) if result.value else CONCIERGE_NAME,
                agent_variable_name="agents",
                history_variable_name="history",
            ),
        )

        user_input = input("Ask a question or type 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break

        await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))
        print(f"# {AuthorRole.USER}: '{user_input}'")

        async for content in chat.invoke():
            print(f"# {content.role} - {content.name or '*'}: '{content.content}'")

        print(f"# IS COMPLETE: {chat.is_complete}")

if __name__ == "__main__":
    asyncio.run(main())
