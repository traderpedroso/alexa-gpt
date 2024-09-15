from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
import ask_sdk_core.utils as ask_utils
import openai
import logging


instructions = """Sou um asistente pessoal"""
mode = "S.P.L"

# Set your OpenAI API key and base URL
openai.api_key = "YOUR_API_KEY"
openai.api_base = "https://api.sostrader.com.br/v1"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""

    def can_handle(self, handler_input):
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        speak_output = f"modo {mode} ativado"
        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["chat_history"] = []

        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


class GptQueryIntentHandler(AbstractRequestHandler):
    """Handler for Gpt Query Intent."""

    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("GptQueryIntent")(handler_input)

    def handle(self, handler_input):
        query = handler_input.request_envelope.request.intent.slots["query"].value
        session_attr = handler_input.attributes_manager.session_attributes
        if "chat_history" not in session_attr:
            session_attr["chat_history"] = []
        response = generate_gpt_response(session_attr["chat_history"], query)
        session_attr["chat_history"].append((query, response))

        return (
            handler_input.response_builder.speak(response)
            .ask("Você pode fazer uma nova pergunta ou falar: sair.")
            .response
        )


class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors."""

    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(exception, exc_info=True)
        speak_output = "Desculpe, tive problemas ao processar seu pedido. Por favor, tente novamente."

        return (
            handler_input.response_builder.speak(speak_output)
            .ask(speak_output)
            .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Single handler for Cancel and Stop Intent."""

    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.CancelIntent")(
            handler_input
        ) or ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input)

    def handle(self, handler_input):
        speak_output = f"Saindo do modo {mode}"

        return handler_input.response_builder.speak(speak_output).response


def generate_gpt_response(chat_history, new_question):
    """Generates a GPT response to a new question"""
    messages = [
        {
            "role": "system",
            "content": f"{instructions}",
        }
    ]
    for question, answer in chat_history[-10:]:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": new_question})

    try:
        response = openai.ChatCompletion.create(
            model="llama-3.1-70b-versatile",
            messages=messages,
            max_tokens=8000,
            temperature=0.7,
        )
        return response.choices[0].message["content"]
    except openai.error.OpenAIError as e:
        logger.error(f"Erro ao gerar resposta: {str(e)}")
        return f"Erro ao gerar resposta: {str(e)}"
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        return f"Erro inesperado: {str(e)}"


sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GptQueryIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
