from typing import Any, Dict

from langchain.base_language import BaseLanguageModel
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.chains import ConversationChain

from real_agents.adapters.executors.base import BaseExecutor
from real_agents.adapters.memory import ConversationBufferMemory


class ChatExecutor(BaseExecutor):
    """Chat Executor."""

    _DEFAULT_TEMPLATE = "The following is a friendly conversation between a human and an AI. \
        The AI is talkative and provides lots of specific details from its context. \
        If the AI does not know the answer to a question, it truthfully says it does not know."
    output_key: str = "result"

    def __init__(self) -> None:
        """Initialize the executor"""
        self.memory = ConversationBufferMemory(return_messages=True)

    def run(
        self,
        user_intent: str,
        llm: BaseLanguageModel,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run the executor.

        Args:
            user_intent: User intent to execute.
            grounding_source: Grounding source to execute the program on.
            llm: Language model to use.
            verbose: Whether to print the logging.

        Returns:
            Result of string.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self._DEFAULT_TEMPLATE),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        method = ConversationChain(
            llm=llm,
            prompt=prompt,
            verbose=verbose,
            memory=self.memory,
        )
        result = method.predict(input=user_intent)
        output = {self.output_key: result}
        return output
from langchain import PromptTemplate

template_base = (
    "{input_string}\nPlease provide {num_questions} natural language questions related to the above contents, "
    "but very different from each other. These questions should be diverse, challenging, "
    "and targeted towards different perspectives. "
    "You should ask these questions like you would ask a human, "
    "but strictly follow the style of your role-playing character.\n"
    "Do not explicitly mention the provided contents; "
    "instead use natural language descriptions for them. "
    "The final result should be a numbered list.".strip() + "\n"
)

QUESTION_SUGGESTION_PROMPT_BASE = PromptTemplate(
    input_variables=["input_string", "num_questions"], template=template_base
)

template_user_profile = (
    "{input_string}\n--------------------\n"
    "{user_description}\n"
    "From now on, you should speak in a style that fully conforms to the given role. \n"
    "Please provide {num_questions} natural language questions related to the above database, "
    "but very different from each other. These questions should be diverse, challenging, "
    "and targeted towards different database tables and columns as well as query types. "
    "You should ask these questions like you would ask a human, "
    "but strictly follow the style of your role-playing character.\n"
    "Do not explicitly mention column or table names in the database; "
    "instead use natural language descriptions for them. "
    "The final result should be a numbered list.".strip() + "\n"
)

QUESTION_SUGGESTION_PROMPT_USER_PROFILE = PromptTemplate(
    input_variables=["input_string", "user_description", "num_questions"], template=template_user_profile
)

template_chat_memory = (
    "{input_string}\n--------------------\n"
    "Here is the conversation between Human and AI.\n"
    "{chat_memory}\n"
    "--------------------\n"
    "Please provide {num_questions} natural language questions related to the above contents, "
    "but very different from each other. These questions should be diverse, challenging, "
    "and targeted towards different perspectives.\n"
    "Keep each questions shorter than 15 words.\n"
    "You should ask these questions like you would ask a human, "
    "but strictly follow the style of your role-playing character.\n"
    "Do not explicitly mention the provided contents; "
    "instead use natural language descriptions for them. "
    "The final result should be a numbered list.".strip() + "\n"
)

QUESTION_SUGGESTION_PROMPT_CHAT_MEMORY = PromptTemplate(
    input_variables=["input_string", "chat_memory", "num_questions"], template=template_chat_memory
)
from typing import Any, Dict

from langchain.base_language import BaseLanguageModel
from langchain.schema import AIMessage, HumanMessage

from real_agents.adapters.memory import ConversationReActBufferMemory
from real_agents.adapters.executors.question_suggestion.chat_memory import QuestionSuggestionChainChatMemory
from real_agents.adapters.executors.question_suggestion.base import QuestionSuggestionChainBase
from real_agents.adapters.executors.question_suggestion.user_profile import QuestionSuggestionChainUserProfile


class QuestionSuggestionExecutor:
    def run(
        self,
        user_intent: str,
        llm: BaseLanguageModel,
        num_questions: int = 3,
        mode: str = "",
        user_profile: str = "",
        chat_memory: ConversationReActBufferMemory = ConversationReActBufferMemory(),
    ) -> Dict[str, Any]:
        if mode == "base":
            method = QuestionSuggestionChainBase.from_prompt(llm)
            inputs = {"input_string": user_intent, "num_questions": num_questions}
        elif mode == "user_profile":
            method = QuestionSuggestionChainUserProfile.from_prompt(llm)
            with open(user_profile) as f:
                inputs = {"input_string": user_intent, "num_questions": num_questions, "user_description": f.read()}
        elif mode == "chat_memory":
            method = QuestionSuggestionChainChatMemory.from_prompt(llm)
            raw_history = chat_memory.load_memory_variables({})["chat_history"]
            refine_history = []
            for msg in raw_history[-4:]:
                if isinstance(msg, HumanMessage):
                    refine_history.append(f"Human: {msg.content}")
                elif isinstance(msg, AIMessage):
                    refine_history.append(f"AI: {msg.content}")
            inputs = {
                "input_string": user_intent,
                "num_questions": num_questions,
                "chat_memory": "\n".join(refine_history),
            }
        else:
            raise ValueError(f"Mode {mode} is not supported")
        return method(inputs)
from __future__ import annotations

from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain

from real_agents.adapters.executors.question_suggestion.base import QuestionSuggestionChainBase
from real_agents.adapters.executors.question_suggestion.prompts import QUESTION_SUGGESTION_PROMPT_USER_PROFILE


class QuestionSuggestionChainUserProfile(QuestionSuggestionChainBase):
    @classmethod
    def from_prompt(cls, llm: BaseLanguageModel) -> QuestionSuggestionChainUserProfile:
        """Load from user profile prompt."""
        llm_chain = LLMChain(llm=llm, prompt=QUESTION_SUGGESTION_PROMPT_USER_PROFILE)
        return cls(llm_chain=llm_chain)
from __future__ import annotations

from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain

from real_agents.adapters.executors.question_suggestion.base import QuestionSuggestionChainBase
from real_agents.adapters.executors.question_suggestion.prompts import QUESTION_SUGGESTION_PROMPT_CHAT_MEMORY


class QuestionSuggestionChainChatMemory(QuestionSuggestionChainBase):
    @classmethod
    def from_prompt(cls, llm: BaseLanguageModel) -> QuestionSuggestionChainChatMemory:
        """Load from user profile prompt."""
        llm_chain = LLMChain(llm=llm, prompt=QUESTION_SUGGESTION_PROMPT_CHAT_MEMORY)
        return cls(llm_chain=llm_chain)
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Extra

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain

from real_agents.adapters.executors.question_suggestion.prompts import QUESTION_SUGGESTION_PROMPT_BASE


class QuestionSuggestionChainBase(Chain, BaseModel):
    """Question Suggestion by Language Models."""

    llm_chain: LLMChain
    output_key: str = "questions"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return self.llm_chain.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return [self.output_key]

    def extract_questions(self, s: str) -> List[str]:
        components = s.split("\n")
        questions = []
        count = 1
        for c in components:
            if c.startswith(f"{count}"):
                questions.append(c.replace(f"{count}.", "").replace(f"{count}", "").strip())
                count += 1
        return questions

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List[str]]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        questions = self.llm_chain.predict(**inputs)
        _run_manager.on_text(questions, color="green", end="\n", verbose=False)
        return {self.output_keys[0]: self.extract_questions(questions)}

    @classmethod
    def from_prompt(cls, llm: BaseLanguageModel) -> QuestionSuggestionChainBase:
        """Load from base prompt."""
        llm_chain = LLMChain(llm=llm, prompt=QUESTION_SUGGESTION_PROMPT_BASE)
        return cls(llm_chain=llm_chain)
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from real_agents.adapters.schema import SQLDatabase


class BaseExecutor(ABC):
    @abstractmethod
    def run(self, user_intent: str, grounding_source: Optional[SQLDatabase]) -> Dict[str, Any]:
        """Run the executor."""
from real_agents.adapters.executors.base import BaseExecutor
from real_agents.adapters.executors.chat_executor import ChatExecutor
from real_agents.adapters.executors.question_suggestion.question_suggestion_executor import (
    QuestionSuggestionExecutor,
    QuestionSuggestionChainChatMemory,
    QuestionSuggestionChainBase,
)
"""Chain that just formats a prompt and calls an LLM."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pydantic import Extra

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.chains.base import Chain
from langchain.input import get_colored_text
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import LLMResult, PromptValue

from real_agents.adapters.data_model import DataModel


class LLMChain(Chain):
    """Chain to run queries against LLMs.

    Example:
        .. code-block:: python

            from langchain import LLMChain, OpenAI, PromptTemplate
            prompt_template = "Tell me a {adjective} joke"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )
            llm = LLMChain(llm=OpenAI(), prompt=prompt)
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        for k, v in inputs.items():
            if isinstance(v, DataModel):
                inputs[k] = v.get_llm_side_data()
        response = self.generate([inputs], run_manager=run_manager)
        return self.create_outputs(response)[0]

    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        from loguru import logger

        logger.trace(
            "\n================<prompt>================\n"
            + prompts[0].to_string()
            + "\n================</prompt>================\n"
        )
        return self.llm.generate_prompt(prompts, stop, callbacks=run_manager.get_child() if run_manager else None)

    async def agenerate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = await self.aprep_prompts(input_list, run_manager=run_manager)
        return await self.llm.agenerate_prompt(
            prompts, stop, callbacks=run_manager.get_child() if run_manager else None
        )

    def prep_prompts(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        """Prepare prompts from inputs."""
        stop = None
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            if run_manager:
                run_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                raise ValueError("If `stop` is present in any inputs, should be present in all.")
            prompts.append(prompt)
        return prompts, stop

    async def aprep_prompts(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        """Prepare prompts from inputs."""
        stop = None
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            if run_manager:
                await run_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                raise ValueError("If `stop` is present in any inputs, should be present in all.")
            prompts.append(prompt)
        return prompts, stop

    def apply(self, input_list: List[Dict[str, Any]], callbacks: Callbacks = None) -> List[Dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        callback_manager = CallbackManager.configure(callbacks, self.callbacks, self.verbose)
        run_manager = callback_manager.on_chain_start(
            {"name": self.__class__.__name__},
            {"input_list": input_list},
        )
        try:
            response = self.generate(input_list, run_manager=run_manager)
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_chain_error(e)
            raise e
        outputs = self.create_outputs(response)
        run_manager.on_chain_end({"outputs": outputs})
        return outputs

    async def aapply(self, input_list: List[Dict[str, Any]], callbacks: Callbacks = None) -> List[Dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        callback_manager = AsyncCallbackManager.configure(callbacks, self.callbacks, self.verbose)
        run_manager = await callback_manager.on_chain_start(
            {"name": self.__class__.__name__},
            {"input_list": input_list},
        )
        try:
            response = await self.agenerate(input_list, run_manager=run_manager)
        except (KeyboardInterrupt, Exception) as e:
            await run_manager.on_chain_error(e)
            raise e
        outputs = self.create_outputs(response)
        await run_manager.on_chain_end({"outputs": outputs})
        return outputs

    def create_outputs(self, response: LLMResult) -> List[Dict[str, str]]:
        """Create outputs from response."""
        return [
            # Get the text of the top generated string.
            {self.output_key: generation[0].text}
            for generation in response.generations
        ]

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        response = await self.agenerate([inputs], run_manager=run_manager)
        return self.create_outputs(response)[0]

    def predict(self, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """Format prompt with kwargs and pass to LLM.

        Args:
            callbacks: Callbacks to pass to LLMChain
            **kwargs: Keys to pass to prompt template.

        Returns:
            Completion from LLM.

        Example:
            .. code-block:: python

                completion = llm.predict(adjective="funny")
        """
        return self(kwargs, callbacks=callbacks)[self.output_key]

    async def apredict(self, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """Format prompt with kwargs and pass to LLM.

        Args:
            callbacks: Callbacks to pass to LLMChain
            **kwargs: Keys to pass to prompt template.

        Returns:
            Completion from LLM.

        Example:
            .. code-block:: python

                completion = llm.predict(adjective="funny")
        """
        return (await self.acall(kwargs, callbacks=callbacks))[self.output_key]

    def predict_and_parse(self, callbacks: Callbacks = None, **kwargs: Any) -> Union[str, List[str], Dict[str, Any]]:
        """Call predict and then parse the results."""
        result = self.predict(callbacks=callbacks, **kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        else:
            return result

    async def apredict_and_parse(
        self, callbacks: Callbacks = None, **kwargs: Any
    ) -> Union[str, List[str], Dict[str, str]]:
        """Call apredict and then parse the results."""
        result = await self.apredict(callbacks=callbacks, **kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        else:
            return result

    def apply_and_parse(
        self, input_list: List[Dict[str, Any]], callbacks: Callbacks = None
    ) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        """Call apply and then parse the results."""
        result = self.apply(input_list, callbacks=callbacks)
        return self._parse_result(result)

    def _parse_result(self, result: List[Dict[str, str]]) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        if self.prompt.output_parser is not None:
            return [self.prompt.output_parser.parse(res[self.output_key]) for res in result]
        else:
            return result

    async def aapply_and_parse(
        self, input_list: List[Dict[str, Any]], callbacks: Callbacks = None
    ) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        """Call apply and then parse the results."""
        result = await self.aapply(input_list, callbacks=callbacks)
        return self._parse_result(result)

    @property
    def _chain_type(self) -> str:
        return "llm_chain"

    @classmethod
    def from_string(cls, llm: BaseLanguageModel, template: str) -> Chain:
        """Create LLMChain from LLM and template."""
        prompt_template = PromptTemplate.from_template(template)
        return cls(llm=llm, prompt=prompt_template)
from typing import Any

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class ExecutorStreamingChainHandler(StreamingStdOutCallbackHandler):
    is_end: bool = False
    _all = []

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """"""
        self._all.append(token)
"""Callback Handler streams to stdout on new llm token."""
import sys
from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class StreamingStdOutCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when LLM errors."""

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when chain errors."""

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
"""Callback Handler streams to stdout on new llm token."""
from typing import Any, Dict, List, Union

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from real_agents.adapters.data_model import DataModel


class JSON_PDA:
    def __init__(self):
        self.stack = []
        self.state = "start"
        self.json = {}
        self.current_key = ""
        self.current_value = ""
        self.escape_next = False

    def transition(self, char):
        if self.escape_next:
            # Add the escaped character to the current key or value and return
            if self.state == "open_key_quote":
                self.current_key += char
            elif self.state == "open_value_quote" or self.state == "open_value_quote_brace":
                self.current_value += char
            self.escape_next = False
            return

        if char == "\\":
            # The next character is an escaped character
            self.escape_next = True
            return

        if self.state == "start":
            if char == "{":
                self.stack.append("{")
                self.state = "open_brace"
            elif char == "`":
                self.state = "open_one_backtick"
                self.stack.append("`")
        elif self.state == "open_one_backtick":
            if char == "`":
                if self.stack[-1] == "`":
                    self.state = "open_two_backticks"
                    self.stack.append("`")
                else:
                    while self.stack.pop() != "`":
                        pass
                    self.state = "start"
            else:
                self.stack.append(char)
        elif self.state == "open_two_backticks":
            if char == "`":
                if self.stack[-1] == "`":
                    self.state = "after_backtick"
                    self.stack.append("`")
                else:
                    while self.stack.pop() != "`":
                        pass
                    self.state = "start"
            else:
                self.stack.append(char)
        elif self.state == "after_backtick":
            if char == "\n":
                self.state = "after_backtick_newline"
        elif self.state == "after_backtick_newline":
            if char == "{":
                self.stack.append("{")
                self.state = "open_brace"
            elif char == "\n":
                self.state = "after_backtick_newline"
            else:
                self.state = "in_block"
        elif self.state == "in_block":
            if char == "`":
                self.stack.pop()
                if len(self.stack) == 0:
                    self.state = "start"
        elif self.state == "open_brace" or self.state == "comma":
            if char == '"':
                self.stack.append('"')
                self.state = "open_key_quote"
                self.current_key = ""
        elif self.state == "open_key_quote" or self.state == "open_value_quote":
            if char != '"':
                if self.state == "open_key_quote":
                    self.current_key += char
                else:
                    self.current_value += char
            else:
                self.stack.pop()
                if self.state == "open_key_quote":
                    self.state = "close_key_quote"
                else:
                    self.state = "close_value_quote"
        elif self.state == "open_value_quote_brace":
            if char == "{":
                self.stack.append("{")
            elif char == "}":
                self.stack.pop()
                if self.stack[-1] == "{" and self.stack[-2] != "{":
                    self.state = "close_value_quote"
            self.current_value += char
        elif self.state == "close_key_quote":
            if char == ":":
                self.state = "after_key"
        elif self.state == "after_key":
            if char == '"':
                self.stack.append('"')
                self.state = "open_value_quote"
                self.current_value = ""
            elif char == "{":
                self.stack.append("{")
                self.state = "open_value_quote_brace"
                self.current_value = "{"
        elif self.state == "close_value_quote":
            self.json[self.current_key] = self.current_value
            if char == ",":
                self.state = "after_value"
            elif char == "}":
                self.stack.pop()
                if len(self.stack) == 0:
                    self.state = "start"
                elif len(self.stack) == 3:
                    self.state = "close_brace"
        elif self.state == "after_value":
            if char == '"':
                self.stack.append('"')
                self.state = "open_key_quote"
        elif self.state == "close_brace":
            if char == "`":
                self.stack.pop()
                if len(self.stack) == 0:
                    self.state = "start"


class AgentStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):
    is_end = False
    generated_tokens: list = []
    for_display: list = []

    # Automata
    pda = JSON_PDA()
    llm_call_id = 0
    _in_json = False
    _in_key = False
    _in_value = False
    _direct_display = True
    _normal_json = False
    json_key: str = ""
    json_tmp_stack: list = []
    action_key_appear = False

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.is_end = False
        self.generated_tokens = []

        self.pda = JSON_PDA()
        self.llm_call_id += 1
        self._in_json = False
        self._in_key = False
        self._in_value = False
        self._direct_display = True
        self._normal_json = False
        self.json_key = ""
        self.json_tmp_stack = []

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """
        Run on new LLM token. Only available when streaming is enabled.
        The tokens that we can decide their types ('plain', 'identifier', 'key', 'action', 'action_input') are stored in `self.for_display`.
        """
        self.generated_tokens.append(token)

        # Automata that monitor json block
        for char in token:
            self.pda.transition(char)

            # Handle the logic of sentences and json blocks
            _type = "plain"

            if self.pda.state in ["open_brace", "open_one_backtick"]:
                self._in_json = True
                self._direct_display = False
                self._normal_json = False
                self.action_key_appear = False

            if self._in_json and not self._normal_json:
                _type = "identifier"

                if self.pda.state == "in_block":
                    _type = "plain"
                    self._normal_json = True

                if self.pda.state == "open_key_quote":
                    if self._in_key:
                        self.json_key += char
                        _type = "key"
                    self._in_key = True
                else:
                    self._in_key = False

                if self.pda.state == "open_value_quote" or self.pda.state == "open_value_quote_brace":
                    if self._in_value:
                        _type = self.json_key
                    self._in_value = True
                else:
                    if self._in_value:
                        self.json_key = ""
                    self._in_value = False

                if self.pda.state == "close_key_quote":
                    # Normal json block

                    if self.json_key not in ["action", "action_input"]:
                        for char_item in self.json_tmp_stack:
                            self.for_display.append(
                                {"text": char_item["text"], "type": "plain", "llm_call_id": self.llm_call_id}
                            )
                        self.json_tmp_stack = []
                        self.for_display.append({"text": char, "type": "plain", "llm_call_id": self.llm_call_id})
                        self._normal_json = True
                        continue

                    else:
                        if self.json_key == "action":
                            self.action_key_appear = True

                        elif self.json_key == "action_input" and self.action_key_appear:
                            # Action json block
                            for char_item in self.json_tmp_stack:
                                char_item["llm_call_id"] = self.llm_call_id
                                self.for_display.append(char_item)
                            self.json_tmp_stack = []
                            self._direct_display = True

            else:
                for char_item in self.json_tmp_stack:
                    self.for_display.append(
                        {"text": char_item["text"], "type": "plain", "llm_call_id": self.llm_call_id}
                    )
                self.json_tmp_stack = []
                self._direct_display = True

            if self.pda.state == "start":
                self._in_json = False

            self.for_display.append(
                {"text": char, "type": _type, "llm_call_id": self.llm_call_id}
            ) if self._direct_display else self.json_tmp_stack.append(
                {"text": char, "type": _type, "llm_call_id": self.llm_call_id}
            )

    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.is_end = True
        for char_item in self.json_tmp_stack:
            self.for_display.append({"text": char_item["text"], "type": "plain", "llm_call_id": self.llm_call_id})

    def on_tool_end(self, output: Union[DataModel, str], **kwargs: Any) -> None:
        """Run on tool end to add observation data model."""
        self.for_display.append({"text": output, "type": "block", "llm_call_id": self.llm_call_id})
"""Base callback handler that can be used to handle callbacks in langchain."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain.schema import AgentAction, AgentFinish, BaseMessage, LLMResult


class LLMManagerMixin:
    """Mixin for LLM callbacks."""

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running."""

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors."""


class ChainManagerMixin:
    """Mixin for chain callbacks."""

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain ends running."""

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain errors."""

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent action."""

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent end."""


class ToolManagerMixin:
    """Mixin for tool callbacks."""

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool ends running."""

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool errors."""


class CallbackManagerMixin:
    """Mixin for callback manager."""

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running."""

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `on_chat_model_start`")

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain starts running."""

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool starts running."""


class RunManagerMixin:
    """Mixin for run manager."""

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on arbitrary text."""


class BaseCallbackHandler(
    LLMManagerMixin,
    ChainManagerMixin,
    ToolManagerMixin,
    CallbackManagerMixin,
    RunManagerMixin,
):
    """Base callback handler that can be used to handle callbacks from langchain."""

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return False

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return False

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return False

    @property
    def ignore_chat_model(self) -> bool:
        """Whether to ignore chat model callbacks."""
        return False


class AsyncCallbackHandler(BaseCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `on_chat_model_start`")

    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""

    async def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""

    async def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""

    async def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""

    async def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""

    async def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on arbitrary text."""

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action."""

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent end."""


class BaseCallbackManager(CallbackManagerMixin):
    """Base callback manager that can be used to handle callbacks from LangChain."""

    def __init__(
        self,
        handlers: List[BaseCallbackHandler],
        inheritable_handlers: Optional[List[BaseCallbackHandler]] = None,
        parent_run_id: Optional[UUID] = None,
    ) -> None:
        """Initialize callback manager."""
        self.handlers: List[BaseCallbackHandler] = handlers
        self.inheritable_handlers: List[BaseCallbackHandler] = inheritable_handlers or []
        self.parent_run_id: Optional[UUID] = parent_run_id

    @property
    def is_async(self) -> bool:
        """Whether the callback manager is async."""
        return False

    def add_handler(self, handler: BaseCallbackHandler, inherit: bool = True) -> None:
        """Add a handler to the callback manager."""
        self.handlers.append(handler)
        if inherit:
            self.inheritable_handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager."""
        self.handlers.remove(handler)
        self.inheritable_handlers.remove(handler)

    def set_handlers(self, handlers: List[BaseCallbackHandler], inherit: bool = True) -> None:
        """Set handlers as the only handlers on the callback manager."""
        self.handlers = []
        self.inheritable_handlers = []
        for handler in handlers:
            self.add_handler(handler, inherit=inherit)

    def set_handler(self, handler: BaseCallbackHandler, inherit: bool = True) -> None:
        """Set handler as the only handler on the callback manager."""
        self.set_handlers([handler], inherit=inherit)
from __future__ import annotations

import asyncio
import functools
import logging
import os
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Generator, List, Optional, Type, TypeVar, Union, cast
from uuid import UUID, uuid4

import langchain
from langchain.callbacks.base import (
    BaseCallbackHandler,
    BaseCallbackManager,
    ChainManagerMixin,
    LLMManagerMixin,
    RunManagerMixin,
    ToolManagerMixin,
)
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.tracers.langchain_v1 import LangChainTracerV1, TracerSessionV1
from langchain.callbacks.tracers.schemas import TracerSession
from langchain.callbacks.tracers.stdout import ConsoleCallbackHandler
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    LLMResult,
    get_buffer_string,
)

logger = logging.getLogger(__name__)
Callbacks = Optional[Union[List[BaseCallbackHandler], BaseCallbackManager]]

openai_callback_var: ContextVar[Optional[OpenAICallbackHandler]] = ContextVar("openai_callback", default=None)
tracing_callback_var: ContextVar[Optional[LangChainTracerV1]] = ContextVar(  # noqa: E501
    "tracing_callback", default=None
)
tracing_v2_callback_var: ContextVar[Optional[LangChainTracer]] = ContextVar(  # noqa: E501
    "tracing_callback_v2", default=None
)


def _get_debug() -> bool:
    return langchain.debug


@contextmanager
def get_openai_callback() -> Generator[OpenAICallbackHandler, None, None]:
    """Get OpenAI callback handler in a context manager."""
    cb = OpenAICallbackHandler()
    openai_callback_var.set(cb)
    yield cb
    openai_callback_var.set(None)


@contextmanager
def tracing_enabled(
    session_name: str = "default",
) -> Generator[TracerSessionV1, None, None]:
    """Get Tracer in a context manager."""
    cb = LangChainTracerV1()
    session = cast(TracerSessionV1, cb.load_session(session_name))
    tracing_callback_var.set(cb)
    yield session
    tracing_callback_var.set(None)


@contextmanager
def tracing_v2_enabled(
    session_name: Optional[str] = None,
    *,
    example_id: Optional[Union[str, UUID]] = None,
    tenant_id: Optional[str] = None,
    session_extra: Optional[Dict[str, Any]] = None,
) -> Generator[TracerSession, None, None]:
    """Get the experimental tracer handler in a context manager."""
    # Issue a warning that this is experimental
    warnings.warn(
        "The experimental tracing v2 is in development. " "This is not yet stable and may change in the future."
    )
    if isinstance(example_id, str):
        example_id = UUID(example_id)
    cb = LangChainTracer(
        tenant_id=tenant_id,
        session_name=session_name,
        example_id=example_id,
        session_extra=session_extra,
    )
    session = cb.ensure_session()
    tracing_v2_callback_var.set(cb)
    yield session
    tracing_v2_callback_var.set(None)


def _handle_event(
    handlers: List[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: Optional[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Generic event handler for CallbackManager."""
    message_strings: Optional[List[str]] = None
    for handler in handlers:
        try:
            if ignore_condition_name is None or not getattr(handler, ignore_condition_name):
                getattr(handler, event_name)(*args, **kwargs)
        except NotImplementedError as e:
            if event_name == "on_chat_model_start":
                if message_strings is None:
                    message_strings = [get_buffer_string(m) for m in args[1]]
                _handle_event(
                    [handler],
                    "on_llm_start",
                    "ignore_llm",
                    args[0],
                    message_strings,
                    *args[2:],
                    **kwargs,
                )
            else:
                logger.warning(f"Error in {event_name} callback: {e}")
        except Exception as e:
            logging.warning(f"Error in {event_name} callback: {e}")


async def _ahandle_event_for_handler(
    handler: BaseCallbackHandler,
    event_name: str,
    ignore_condition_name: Optional[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    try:
        if ignore_condition_name is None or not getattr(handler, ignore_condition_name):
            event = getattr(handler, event_name)
            if asyncio.iscoroutinefunction(event):
                await event(*args, **kwargs)
            else:
                await asyncio.get_event_loop().run_in_executor(None, functools.partial(event, *args, **kwargs))
    except NotImplementedError as e:
        if event_name == "on_chat_model_start":
            message_strings = [get_buffer_string(m) for m in args[1]]
            await _ahandle_event_for_handler(
                handler,
                "on_llm_start",
                "ignore_llm",
                args[0],
                message_strings,
                *args[2:],
                **kwargs,
            )
        else:
            logger.warning(f"Error in {event_name} callback: {e}")
    except Exception as e:
        logger.warning(f"Error in {event_name} callback: {e}")


async def _ahandle_event(
    handlers: List[BaseCallbackHandler],
    event_name: str,
    ignore_condition_name: Optional[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Generic event handler for AsyncCallbackManager."""
    await asyncio.gather(
        *(
            _ahandle_event_for_handler(handler, event_name, ignore_condition_name, *args, **kwargs)
            for handler in handlers
        )
    )


BRM = TypeVar("BRM", bound="BaseRunManager")


class BaseRunManager(RunManagerMixin):
    """Base class for run manager (a bound callback manager)."""

    def __init__(
        self,
        run_id: UUID,
        handlers: List[BaseCallbackHandler],
        inheritable_handlers: List[BaseCallbackHandler],
        parent_run_id: Optional[UUID] = None,
    ) -> None:
        """Initialize run manager."""
        self.run_id = run_id
        self.handlers = handlers
        self.inheritable_handlers = inheritable_handlers
        self.parent_run_id = parent_run_id

    @classmethod
    def get_noop_manager(cls: Type[BRM]) -> BRM:
        """Return a manager that doesn't perform any operations."""
        return cls(uuid4(), [], [])


class RunManager(BaseRunManager):
    """Sync Run Manager."""

    def on_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> Any:
        """Run when text is received."""
        _handle_event(
            self.handlers,
            "on_text",
            None,
            text,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class AsyncRunManager(BaseRunManager):
    """Async Run Manager."""

    async def on_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> Any:
        """Run when text is received."""
        await _ahandle_event(
            self.handlers,
            "on_text",
            None,
            text,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class CallbackManagerForLLMRun(RunManager, LLMManagerMixin):
    """Callback manager for LLM run."""

    def on_llm_new_token(
        self,
        token: str,
        **kwargs: Any,
    ) -> None:
        """Run when LLM generates a new token."""
        _handle_event(
            self.handlers,
            "on_llm_new_token",
            "ignore_llm",
            token=token,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        _handle_event(
            self.handlers,
            "on_llm_end",
            "ignore_llm",
            response,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""
        _handle_event(
            self.handlers,
            "on_llm_error",
            "ignore_llm",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class AsyncCallbackManagerForLLMRun(AsyncRunManager, LLMManagerMixin):
    """Async callback manager for LLM run."""

    async def on_llm_new_token(
        self,
        token: str,
        **kwargs: Any,
    ) -> None:
        """Run when LLM generates a new token."""
        await _ahandle_event(
            self.handlers,
            "on_llm_new_token",
            "ignore_llm",
            token,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        await _ahandle_event(
            self.handlers,
            "on_llm_end",
            "ignore_llm",
            response,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    async def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""
        await _ahandle_event(
            self.handlers,
            "on_llm_error",
            "ignore_llm",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class CallbackManagerForChainRun(RunManager, ChainManagerMixin):
    """Callback manager for chain run."""

    def get_child(self) -> CallbackManager:
        """Get a child callback manager."""
        manager = CallbackManager([], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)
        return manager

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        _handle_event(
            self.handlers,
            "on_chain_end",
            "ignore_chain",
            outputs,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""
        _handle_event(
            self.handlers,
            "on_chain_error",
            "ignore_chain",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run when agent action is received."""
        _handle_event(
            self.handlers,
            "on_agent_action",
            "ignore_agent",
            action,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run when agent finish is received."""
        _handle_event(
            self.handlers,
            "on_agent_finish",
            "ignore_agent",
            finish,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class AsyncCallbackManagerForChainRun(AsyncRunManager, ChainManagerMixin):
    """Async callback manager for chain run."""

    def get_child(self) -> AsyncCallbackManager:
        """Get a child callback manager."""
        manager = AsyncCallbackManager([], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)
        return manager

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        await _ahandle_event(
            self.handlers,
            "on_chain_end",
            "ignore_chain",
            outputs,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    async def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""
        await _ahandle_event(
            self.handlers,
            "on_chain_error",
            "ignore_chain",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run when agent action is received."""
        await _ahandle_event(
            self.handlers,
            "on_agent_action",
            "ignore_agent",
            action,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run when agent finish is received."""
        await _ahandle_event(
            self.handlers,
            "on_agent_finish",
            "ignore_agent",
            finish,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class CallbackManagerForToolRun(RunManager, ToolManagerMixin):
    """Callback manager for tool run."""

    def get_child(self) -> CallbackManager:
        """Get a child callback manager."""
        manager = CallbackManager([], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)
        return manager

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        _handle_event(
            self.handlers,
            "on_tool_end",
            "ignore_agent",
            output,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""
        _handle_event(
            self.handlers,
            "on_tool_error",
            "ignore_agent",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    def on_tool_end_data_model(
        self,
        output,
        **kwargs: Any,
    ):
        """Return the data model for the on_tool_end event."""
        _handle_event(
            self.handlers,
            "on_tool_end_data_model",
            "ignore_agent",
            output,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class AsyncCallbackManagerForToolRun(AsyncRunManager, ToolManagerMixin):
    """Async callback manager for tool run."""

    def get_child(self) -> AsyncCallbackManager:
        """Get a child callback manager."""
        manager = AsyncCallbackManager([], parent_run_id=self.run_id)
        manager.set_handlers(self.inheritable_handlers)
        return manager

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        await _ahandle_event(
            self.handlers,
            "on_tool_end",
            "ignore_agent",
            output,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

    async def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""
        await _ahandle_event(
            self.handlers,
            "on_tool_error",
            "ignore_agent",
            error,
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )


class CallbackManager(BaseCallbackManager):
    """Callback manager that can be used to handle callbacks from langchain."""

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> CallbackManagerForLLMRun:
        """Run when LLM starts running."""
        if run_id is None:
            run_id = uuid4()

        _handle_event(
            self.handlers,
            "on_llm_start",
            "ignore_llm",
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

        return CallbackManagerForLLMRun(run_id, self.handlers, self.inheritable_handlers, self.parent_run_id)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> CallbackManagerForLLMRun:
        """Run when LLM starts running."""
        if run_id is None:
            run_id = uuid4()
        _handle_event(
            self.handlers,
            "on_chat_model_start",
            "ignore_chat_model",
            serialized,
            messages,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

        # Re-use the LLM Run Manager since the outputs are treated
        # the same for now
        return CallbackManagerForLLMRun(run_id, self.handlers, self.inheritable_handlers, self.parent_run_id)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> CallbackManagerForChainRun:
        """Run when chain starts running."""
        if run_id is None:
            run_id = uuid4()

        _handle_event(
            self.handlers,
            "on_chain_start",
            "ignore_chain",
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

        return CallbackManagerForChainRun(run_id, self.handlers, self.inheritable_handlers, self.parent_run_id)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> CallbackManagerForToolRun:
        """Run when tool starts running."""
        if run_id is None:
            run_id = uuid4()

        _handle_event(
            self.handlers,
            "on_tool_start",
            "ignore_agent",
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

        return CallbackManagerForToolRun(run_id, self.handlers, self.inheritable_handlers, self.parent_run_id)

    @classmethod
    def configure(
        cls,
        inheritable_callbacks: Callbacks = None,
        local_callbacks: Callbacks = None,
        verbose: bool = False,
    ) -> CallbackManager:
        """Configure the callback manager."""
        return _configure(cls, inheritable_callbacks, local_callbacks, verbose)


class AsyncCallbackManager(BaseCallbackManager):
    """Async callback manager that can be used to handle callbacks from LangChain."""

    @property
    def is_async(self) -> bool:
        """Return whether the handler is async."""
        return True

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForLLMRun:
        """Run when LLM starts running."""
        if run_id is None:
            run_id = uuid4()

        await _ahandle_event(
            self.handlers,
            "on_llm_start",
            "ignore_llm",
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

        return AsyncCallbackManagerForLLMRun(run_id, self.handlers, self.inheritable_handlers, self.parent_run_id)

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if run_id is None:
            run_id = uuid4()

        await _ahandle_event(
            self.handlers,
            "on_chat_model_start",
            "ignore_chat_model",
            serialized,
            messages,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

        return AsyncCallbackManagerForLLMRun(run_id, self.handlers, self.inheritable_handlers, self.parent_run_id)

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForChainRun:
        """Run when chain starts running."""
        if run_id is None:
            run_id = uuid4()

        await _ahandle_event(
            self.handlers,
            "on_chain_start",
            "ignore_chain",
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

        return AsyncCallbackManagerForChainRun(run_id, self.handlers, self.inheritable_handlers, self.parent_run_id)

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> AsyncCallbackManagerForToolRun:
        """Run when tool starts running."""
        if run_id is None:
            run_id = uuid4()

        await _ahandle_event(
            self.handlers,
            "on_tool_start",
            "ignore_agent",
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=self.parent_run_id,
            **kwargs,
        )

        return AsyncCallbackManagerForToolRun(run_id, self.handlers, self.inheritable_handlers, self.parent_run_id)

    @classmethod
    def configure(
        cls,
        inheritable_callbacks: Callbacks = None,
        local_callbacks: Callbacks = None,
        verbose: bool = False,
    ) -> AsyncCallbackManager:
        """Configure the callback manager."""
        return _configure(cls, inheritable_callbacks, local_callbacks, verbose)


T = TypeVar("T", CallbackManager, AsyncCallbackManager)


def _configure(
    callback_manager_cls: Type[T],
    inheritable_callbacks: Callbacks = None,
    local_callbacks: Callbacks = None,
    verbose: bool = False,
) -> T:
    """Configure the callback manager."""
    callback_manager = callback_manager_cls([])
    if inheritable_callbacks or local_callbacks:
        if isinstance(inheritable_callbacks, list) or inheritable_callbacks is None:
            inheritable_callbacks_ = inheritable_callbacks or []
            callback_manager = callback_manager_cls(
                handlers=inheritable_callbacks_.copy(),
                inheritable_handlers=inheritable_callbacks_.copy(),
            )
        else:
            callback_manager = callback_manager_cls(
                handlers=inheritable_callbacks.handlers,
                inheritable_handlers=inheritable_callbacks.inheritable_handlers,
                parent_run_id=inheritable_callbacks.parent_run_id,
            )
        local_handlers_ = (
            local_callbacks
            if isinstance(local_callbacks, list)
            else (local_callbacks.handlers if local_callbacks else [])
        )
        for handler in local_handlers_:
            callback_manager.add_handler(handler, False)

    tracer = tracing_callback_var.get()
    open_ai = openai_callback_var.get()
    tracing_enabled_ = (
        os.environ.get("LANGCHAIN_TRACING") is not None
        or tracer is not None
        or os.environ.get("LANGCHAIN_HANDLER") is not None
    )

    tracer_v2 = tracing_v2_callback_var.get()
    tracing_v2_enabled_ = os.environ.get("LANGCHAIN_TRACING_V2") is not None or tracer_v2 is not None
    tracer_session = os.environ.get("LANGCHAIN_SESSION")
    debug = _get_debug()
    if tracer_session is None:
        tracer_session = "default"
    if verbose or debug or tracing_enabled_ or tracing_v2_enabled_ or open_ai is not None:
        if verbose and not any(isinstance(handler, StdOutCallbackHandler) for handler in callback_manager.handlers):
            if debug:
                pass
            else:
                callback_manager.add_handler(StdOutCallbackHandler(), False)
        if debug and not any(isinstance(handler, ConsoleCallbackHandler) for handler in callback_manager.handlers):
            callback_manager.add_handler(ConsoleCallbackHandler(), True)
        if tracing_enabled_ and not any(
            isinstance(handler, LangChainTracerV1) for handler in callback_manager.handlers
        ):
            if tracer:
                callback_manager.add_handler(tracer, True)
            else:
                handler = LangChainTracerV1()
                handler.load_session(tracer_session)
                callback_manager.add_handler(handler, True)
        if tracing_v2_enabled_ and not any(
            isinstance(handler, LangChainTracer) for handler in callback_manager.handlers
        ):
            if tracer_v2:
                callback_manager.add_handler(tracer_v2, True)
            else:
                try:
                    handler = LangChainTracer(session_name=tracer_session)
                    handler.ensure_session()
                    callback_manager.add_handler(handler, True)
                except Exception as e:
                    logger.debug("Unable to load requested LangChainTracer", e)
        if open_ai is not None and not any(
            isinstance(handler, OpenAICallbackHandler) for handler in callback_manager.handlers
        ):
            callback_manager.add_handler(open_ai, True)
    return callback_manager
from real_agents.adapters.callbacks.agent_streaming import JSON_PDA, AgentStreamingStdOutCallbackHandler
from real_agents.adapters.callbacks.base import BaseCallbackHandler, BaseCallbackManager, AsyncCallbackHandler
from real_agents.adapters.callbacks.executor_streaming import ExecutorStreamingChainHandler
from real_agents.adapters.callbacks.manager import CallbackManager, CallbackManagerForChainRun
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from real_agents.adapters.data_model.base import DataModel
from real_agents.adapters.data_model.templates.skg_templates.database_templates import serialize_db
from real_agents.adapters.data_model.templates.skg_templates.table_templates import serialize_df
import json


class KaggleDataModel(DataModel):
    """A data model for KaggleDataModel.
    We only support the csv and sqlite format for now.
    raw_data is a Dict[str, TableDataModel]
    raw_data_path is List[str]
    raw_data_name is Dict[str, str]
    """

    def get_llm_side_data(self, serialize_method: str = "tsv", num_visible_rows: int = 3) -> Any:
        formatted_tables = []
        for _raw_data_path in self.raw_data_path:
            table_data = self.raw_data[_raw_data_path]
            table_name = self.raw_data_name[_raw_data_path]
            table_path = _raw_data_path
            formatted_table = serialize_df(table_data, table_name, table_path, serialize_method, num_visible_rows)
            formatted_tables.append(formatted_table)
        return "\n".join(formatted_tables)

    @staticmethod
    def to_react_table(table: pd.DataFrame) -> str:
        columns = list(map(lambda item: {"accessorKey": item, "header": item}, table.columns.tolist()))
        # FIXME: NaN may not be handled here.
        data = table.fillna("").to_dict(orient="records")
        table = json.dumps({"columns": columns, "data": data})
        return table

    def get_human_side_data(self) -> Any:
        # In the frontend, we show the first few rows of each table
        react_tables = {}
        for table_path in self.raw_data_path:
            table_name = self.raw_data_name[table_path]
            table = self.raw_data[table_path]
            react_tables[table_name] = self.to_react_table(table)
        return json.dumps(react_tables)
from typing import Any

from real_agents.adapters.data_model.base import DataModel


class ImageDataModel(DataModel):
    """A data model for image."""

    simple_filename = ""

    def get_raw_data(self) -> Any:
        return self.raw_data

    def get_llm_side_data(self) -> Any:
        if self.simple_filename == "":
            import os

            self.simple_filename = os.path.basename(self.raw_data_path)
        string = "image: " + self.simple_filename
        return string

    def get_human_side_data(self) -> Any:
        return self.raw_data["base64_string"]
import json
from bs4 import BeautifulSoup
from collections import defaultdict
from typing import Any, Dict, List, Union
from real_agents.adapters.data_model.base import DataModel
import requests
import re
import tiktoken

JsonNode = Dict[str, Union[str, List[Any], int]]
PossibleTemplate = Dict[str, Union[str, List[Any], int]]
OptimizedTemplate = Dict[str, Union[str, List[Any], int, set]]
PossibleTemplates = Dict[str, PossibleTemplate]


def find_potential_templates(node, possible_templates):
    """Find all potential templates in the HTML tree."""
    if node.name:  # Element node
        attributes = {attr: node[attr] for attr in node.attrs}
        children = []
        for child in node.children:
            child_json = find_potential_templates(child, possible_templates)
            if child_json:
                children.append(child_json)

        # Max depth of the tree
        depth = max([c["depth"] for c in children], default=0) + 1

        # Create a template hash
        template_hash = f"{node.name}#{sorted(attributes.keys())}#{[c['template_hash'] for c in children]}"

        # Gather template values
        template_values = list(attributes.values()) + [val for c in children for val in c["template_values"]]

        json_node = {
            "type": "ELEMENT",
            "tag_name": node.name,
            "attributes": attributes,
            "children": children,
            "template_hash": template_hash,
            "template_values": template_values,
            "depth": depth,
        }

        # Add node to possible templates
        if template_hash in possible_templates:
            if possible_templates[template_hash][0]["depth"] != depth:
                raise ValueError(f"Template depth mismatch for template {template_hash}")
            possible_templates[template_hash].append(json_node)
        else:
            possible_templates[template_hash] = [json_node]

        return json_node
    elif isinstance(node, str):  # Text node
        text = node.strip()
        if text:
            return {"type": "TEXT", "content": text, "template_hash": "TEXT", "template_values": [text], "depth": 0}
    return None


def optimize_template(template):
    """Check and adjust the template in possible_templates to optimize style."""
    values_to_inline = {
        i
        for i in range(len(template["nodes"][0]["templateValues"]))
        if all(n["templateValues"][i] == template["nodes"][0]["templateValues"][i] for n in template["nodes"])
    }
    return {**template, "valuesToInline": values_to_inline}


def is_string_a_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_placeholder(template, value_index):
    """Get the placeholder for the value at the given index in the template."""
    placeholder_index = value_index + 1 - len([i for i in template["valuesToInline"] if i < value_index])
    return f"${placeholder_index}"


def create_template_tree(node, templates, render_for_template, current_value_index=0):
    """Convert the DOM into processed template tree."""
    if node["type"] == "TEXT":
        if current_value_index in render_for_template["valuesToInline"]:
            return {
                "template": node["content"],
                "valueIndex": current_value_index + 1,
                "consumedTemplates": [node["templateHash"]],
            }
        else:
            return {
                "template": get_placeholder(render_for_template, current_value_index),
                "valueIndex": current_value_index + 1,
                "consumedTemplates": [node["templateHash"]],
            }

    else:
        updated_value_index = current_value_index
        consumed_templates = [node["templateHash"]]

        attrs = "".join(
            [
                f' {k}="{v}"'
                if updated_value_index + i in render_for_template["valuesToInline"]
                else f" {k}={get_placeholder(render_for_template, updated_value_index + i)}"
                for i, (k, v) in enumerate(node["attributes"].items())
            ]
        )
        updated_value_index += len(node["attributes"])

        children = []
        for child in node["children"]:
            child_template = create_template_tree(child, templates, render_for_template, updated_value_index)
            children.append(child_template["template"])
            updated_value_index = child_template["valueIndex"]
            consumed_templates.extend(child_template["consumedTemplates"])

        return {
            "template": f"<{node['tagName'].lower()}{attrs}/>"
            if not children
            else f"<{node['tagName'].lower()}{attrs}>{''.join(children)}</{node['tagName'].lower()}>",
            "valueIndex": updated_value_index,
            "consumedTemplates": consumed_templates,
        }


def serialize_tree(node, templates):
    """Serialize the template tree into HTML string."""
    if node["type"] == "TEXT":
        return node["content"]
    elif node["templateHash"] in templates:
        template = templates[node["templateHash"]]
        return f"{{T{template['label']}({','.join([str(v) if is_string_a_number(v) else json.dumps(v) for i, v in enumerate(node['templateValues']) if i not in template['valuesToInline']])})}}"
    else:
        attrs = "".join([f' {k}="{v}"' for k, v in node["attributes"].items()])
        children = "".join([serialize_tree(c, templates) for c in node["children"]])
        return (
            f"<{node['tagName'].lower()}{attrs}/>"
            if not children
            else f"<{node['tagName'].lower()}{attrs}>{children}</{node['tagName'].lower()}>"
        )


def truncate_html_by_tokens(html_string, max_tokens, model_name, num_tags_to_remove_each_time=10):
    tokens_count = count_tokens(html_string, model_name)
    num_tags_to_remove_each_time = round(tokens_count / 500)
    soup = BeautifulSoup(html_string, "html.parser")
    # Remove all iframe tags
    html_string = remove_iframes(html_string)
    while tokens_count > max_tokens:
        tags = soup.find_all(True)  # find all tags
        # remove the last N tags
        for tag in tags[-num_tags_to_remove_each_time:]:
            tag.decompose()

        html_string = str(soup)

        # re-count the tokens
        tokens_count = count_tokens(html_string, model_name)

    return html_string


# hacky way
def remove_iframes(html_string):
    # Remove all iframe tags using regex
    return re.sub("<iframe.*?/iframe>", "", html_string, flags=re.DOTALL)


# if you wanna change encoding schema, refer to https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def count_tokens(text, model_name):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


class HTMLDataModel(DataModel):
    """A data model for HTML, for webot purpose."""

    def get_llm_side_data(self) -> str:
        html_string = self.raw_data
        truncated_html_string = truncate_html_by_tokens(html_string, 5000, "gpt-4")
        return truncated_html_string
from copy import deepcopy
from typing import Any, Dict


def convert(_input_json: Dict[str, Any]) -> Dict[str, Any]:
    input_json = deepcopy(_input_json)
    assert isinstance(input_json["out"], list)

    input_json["out"] = input_json["out"][:5]
    extracted_keys = [
        "body",
        "title",
        "created",
        "url",
        "tags",
    ]
    input_json["out"] = [{k: r[k] for k in extracted_keys if k in r} for r in input_json["out"]]
    return input_json
from copy import deepcopy
from typing import Any, Dict


def convert(_input_json: Dict[str, Any]) -> Dict[str, Any]:
    input_json = deepcopy(_input_json)
    assert isinstance(input_json["out"], list)

    input_json["out"]["articles"] = input_json["out"]["articles"][:5]
    return input_json
from __future__ import annotations

import json
import os
from typing import Any, Dict

import yaml
from prance import ResolvingParser
from pydantic import BaseModel

# get the absolute path of the current file
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))


class APIYamlModel(BaseModel):
    info: Dict

    @classmethod
    def from_yaml(cls, yaml_path: str) -> APIYamlModel:
        return cls(info=APIYamlModel.yaml_to_json(yaml_path))

    @classmethod
    def from_json(cls, json_path: str) -> APIYamlModel:
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
        return cls(info=json_data)

    def to_yaml(self) -> Dict:
        yaml_data = yaml.safe_dump(self.info, sort_keys=False)
        return yaml_data

    def to_json(self) -> Dict:
        return self.info

    @staticmethod
    def yaml_to_json(yaml_path: str) -> Dict:
        # Open the OpenAPI YAML file
        # Load the YAML contents into a Python dictionary
        # json_data = yaml.safe_load(yaml_file)
        # there are #/xxxx/yyyy reference in openapi.yaml
        parsed = ResolvingParser(yaml_path, backend="openapi-spec-validator", strict=False)
        json_data = json.loads(parsed.json())
        return json_data

    @staticmethod
    def json_to_yaml(json_path: str) -> Any:
        # Open the OpenAPI JSON file
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
            yaml_data = yaml.dump(json_data)
            return yaml_data
import os
from typing import Any, Dict
import importlib.util
import tiktoken

from real_agents.adapters.data_model.plugin.base import APIYamlModel
from real_agents.adapters.data_model.utils import indent_multiline_string


def import_function_from_file(filepath, function_name):
    spec = importlib.util.spec_from_file_location("module.name", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, function_name)

    return function


def process_one_param(param_dict: Dict[str, Any]) -> str:
    name = param_dict.get("name", None)
    description = param_dict.get("description", None)
    required = param_dict.get("required", None)

    schema = param_dict.get("schema", {})
    type = schema.get("type", "UnknownType")
    value_choices = schema.get("enum", [])

    ret = (
        f"`{name}` ({type}, {'required' if required else 'optional'}): {description}."
        f"{'Examples:' + ','.join([str(_) for _ in value_choices]) if len(value_choices) > 0 else ''}"
    )
    return ret


def process_one_property(name: str, value_dict: Dict[str, Any]) -> str:
    description = value_dict.get("description", None)
    required = value_dict.get("required", None)

    type = value_dict.get("type", "UnknownType")
    value_choices = value_dict.get("enum", [])

    ret = (
        f"`{name}` ({type}, {'required' if required else 'optional'}): {description}."
        f"{'Examples:' + ','.join(value_choices) if len(value_choices) > 0 else ''}"
    )
    return ret


class SpecModel:
    def __init__(self, yaml_path: str, model_name: str = "gpt-4") -> None:
        # fixme: Must move out the logic of yaml path
        self.yaml_path = yaml_path
        self.full_spec = APIYamlModel.from_yaml(yaml_path).to_json()
        self.paths = self.full_spec["paths"]

        # Process the description
        enc = tiktoken.encoding_for_model(model_name)
        if "description" in self.full_spec["info"]:
            if len(self.full_spec["info"]["description"]) > 200:
                self.full_spec["info"]["description"] = enc.decode(
                    enc.encode(self.full_spec["info"]["description"])[:200]
                )

    def load_personnel_info(self, api_key: str):
        # Get the dir of the yaml file
        yaml_dir = os.path.dirname(self.yaml_path)
        personnel_load_dir = os.path.join(yaml_dir, "personnel.py")

        if not os.path.exists(personnel_load_dir):
            return {}, {}

        # Reload openapi.yaml
        reload_openapi = import_function_from_file(personnel_load_dir, "reload_openapi")
        resolved_json, new_paths_json = reload_openapi(api_key, self.full_spec)
        self.full_spec = resolved_json
        self.full_spec["info"] = resolved_json["info"]
        self.paths = resolved_json["paths"]

        # Reload the endpoints functions
        reload_endpoints = import_function_from_file(personnel_load_dir, "reload_endpoints")
        new_endpoint2caller = reload_endpoints(new_paths_json)

        # Reload the endpoints datamodels
        # todo: Add reload datamodels function
        new_endpoints2output_model = {k: lambda x: x for k in new_paths_json}

        return new_endpoint2caller, new_endpoints2output_model

    def prepare_spec(self, include_params: bool = True) -> str:
        path_names = list(self.paths.keys())
        ret = self.prepare_spec_for_one_path(path_names[0], include_params=include_params)

        if len(path_names) > 1:
            ret += "\n"

        for path_name in path_names[1:]:
            ret += (
                self.prepare_spec_for_one_path(path_name, include_api_info=False, include_params=include_params) + "\n"
            )

        return ret

    def list_endpoints(self) -> str:
        ret = ""
        for ep, ep_spec in self.paths.items():
            assert len(ep_spec) == 1, "Support two request methods!"
            request_method = list(ep_spec.keys())[0]
            func_spec = ep_spec[request_method]
            desc = func_spec.get("summary", None)
            ret += f"`{ep}`: {desc}\n"
        return ret.strip()

    def prepare_spec_for_one_path(
        self,
        path_name: str,
        include_api_info: bool = True,
        include_params: bool = True,
    ) -> str:
        func_dict = self.paths[path_name]
        if "servers" in func_dict:
            del func_dict["servers"]

        rets = []
        for request_method in list(func_dict.keys()):
            candidate_inputs_str = ""
            func_spec = func_dict[request_method]

            # Only GET and DELETE are processed, others are not properly processed
            if request_method.lower() not in ["get", "post", "put", "patch", "delete"]:
                raise ValueError("Unknown request method")

            # TODO: not sure how to arrange input when post method has "parameters"
            func_summary = func_spec.get("summary", None)
            func_description = func_spec.get("description", None)
            candidate_inputs = func_spec.get("parameters", [])
            candidate_inputs_str += "\n".join(process_one_param(p) for p in candidate_inputs)

            if request_method.lower() == "post" and "requestBody" in func_spec:
                request_body = func_spec["requestBody"]
                assert "content" in request_body, "Must have content in requestBody"
                content_dict = request_body["content"]
                assert len(content_dict) == 1, "Support one content type"
                content_type = list(content_dict.keys())[0]
                content = content_dict[content_type]
                assert "schema" in content, "Must have schema in requestBody"
                if "properties" in content["schema"]:
                    properties = content["schema"]["properties"]
                    candidate_inputs_str += "\n".join(process_one_property(n, vd) for n, vd in properties.items())

            ret = ""
            if include_api_info:
                ret += f"""Name: {self.full_spec["info"]["title"]}\n{'Description: ' + self.full_spec["info"]['description'] if
                "description" in self.full_spec["info"] else ""}\n"""

            ret += f"""\tSummary: {func_summary}\n"""
            ret += f"""\tDescription: {func_description}\n"""
            candidate_inputs_str = "None" if len(candidate_inputs_str) == 0 else candidate_inputs_str
            ret += (
                f"""\tInput: \n{indent_multiline_string(candidate_inputs_str, indent=2)}\n""" if include_params else ""
            )
            rets.append(ret)

        return f"""Endpoint: {path_name}\n""" + "\n".join(rets)
from copy import deepcopy
from typing import Any, Dict


def convert(_input_json: Dict[str, Any]) -> Dict[str, Any]:
    input_json = deepcopy(_input_json)
    assert isinstance(input_json["out"], list)

    input_json["out"] = input_json["out"][:5]

    for i, job in enumerate(input_json["out"]):
        cleaned_job_item = input_json["out"][i]
        del cleaned_job_item["id"]
        del cleaned_job_item["created"]
        input_json["out"][i] = cleaned_job_item

    return input_json
from real_agents.adapters.data_model.plugin.spec import APIYamlModel, SpecModel
import re
import textwrap
from typing import List, Dict, Any, Optional
from langchain.schema import BaseMessage
import tiktoken

# format of agent action
ACTION_FORMAT = """```json
{{
    "action": "{_action}",
    "action_input": "{_action_input}",
}}
```"""

# format of tool call(code) & tool output(response)
TOOL_FORMAT = {
    "code": """<code>
{_intermediate_steps}
</code>

<output>
{_result}
</output>
""",
    "plugin": """<plugin_call>
{_intermediate_steps}
</plugin_call>

<output>
{_result}
</output>
""",
}

# format to wrap tool call + tool output together
TOOL_RESPONSE_FORMAT = """[RESPONSE_BEGIN]
{_response}
[RESPONSE_END]
"""


class MessageDataModel:
    """A data model for Message Management, general purpose."""

    @staticmethod
    def _count_tokens(test_string: str) -> int:
        """copy of langchain _get_num_token_default_method"""
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = len(enc.encode(test_string))
        return tokens

    @classmethod
    def _get_num_tokens_from_messages(cls, buffer: List[BaseMessage]) -> int:
        return sum([cls._count_tokens(m.content) for m in buffer])

    @classmethod
    def truncate_text(cls, raw_text: str, max_token: Optional[int] = 250, trunc_ratio: int = 0.5) -> str:
        """heuristic truncation for single long string & code"""
        tokens = cls._count_tokens(raw_text)
        if max_token is None or tokens <= max_token:
            return raw_text

        # assume we keep the first ratio * max_tokens and the (1 - ratio) * max_tokens
        half_tokens = int(max_token * trunc_ratio)
        lines = raw_text.strip().split("\n")
        lines = [" ".join(line.split(" ")[:100]) for line in lines]
        total_lines = len(lines)

        # first half
        left = 0
        right = total_lines // 2
        while left < right:
            mid = (left + right) >> 1
            text = "\n".join(lines[0:mid])
            token = cls._count_tokens(text)
            if token > half_tokens:
                right = mid
            else:
                left = mid + 1
        first_half = "\n".join(lines[0:right])

        # last half
        left = total_lines // 2 + 1
        right = total_lines - 1
        while left < right:
            mid = (left + right) >> 1
            text = "\n".join(lines[mid:])
            token = cls._count_tokens(text)
            if token > half_tokens:
                right = mid
            else:
                left = mid + 1
        second_half = "\n".join(lines[left:])

        if first_half != "" or second_half != "":
            return f"{first_half}\n...\n[too long to show]\n...\n{second_half}"
        else:
            # if len(first_half_list) == 0 and len(last_half_list) == 0:
            # if all lines >= max_token, return last 100 words as truncated results.
            return f"...\n[too long to show]\n...\n{raw_text[-100:]}"

    @classmethod
    def truncate_chat_history(cls, full_inputs: Dict[str, Any], max_token: int = 2500) -> Dict[str, Any]:
        _input = full_inputs["input"]
        agent_scratchpad = full_inputs["agent_scratchpad"]
        agent_scratchpad = "\n".join([_.content for _ in agent_scratchpad])
        _input_tokens = cls._count_tokens(_input)
        _scratchpad_tokens = cls._count_tokens(agent_scratchpad)

        left_tokens = max_token - _scratchpad_tokens - _input_tokens
        chat_history = full_inputs["chat_history"]

        curr_buffer_length = cls._get_num_tokens_from_messages(chat_history)
        while len(chat_history) != 0 and curr_buffer_length > left_tokens:
            chat_history.pop(0)
            curr_buffer_length = cls._get_num_tokens_from_messages(chat_history)
        full_inputs["chat_history"] = chat_history
        return full_inputs

    @staticmethod
    def _extract_value(json_string: str, key: str) -> str:
        pattern = re.compile(rf'"?{key}"?\s*:\s*("((?:[^"\\]|\\.)*)"|(\b[^,\s]*\b))', re.MULTILINE)
        match = pattern.search(json_string)
        if match:
            result = match.group(1).replace('\\"', '"').replace("\\\\", "\\").strip('"').strip("'").strip()
            # result = f"\"{result}\""
            return result
        raise ValueError(f"Could not find {key} in {json_string}")

    @staticmethod
    def _extract_response(
        chat_history: str,
        begin_marker: str = "[RESPONSE_BEGIN]",
        end_marker: str = "[RESPONSE_END]",
        ai_msg_marker: str = "AI:",
    ):
        code_blocks = chat_history.split(ai_msg_marker)
        pattern = r"\[RESPONSE_BEGIN\](.*?)\[RESPONSE_END\]"

        cleaned_output = []
        for code_block in code_blocks:
            matches = re.findall(pattern, code_block, re.DOTALL)
            if matches:
                cleaned_output.append(matches[0].strip())
        return "\n".join(cleaned_output)

    @classmethod
    def extract_action_for_llm(cls, text, max_token: int = 500) -> str:
        """Since Action should be fully inputted into an Agent, so we do not perform truncation here."""
        action_format = ACTION_FORMAT
        cleaned_output = text.strip()
        try:
            _action = cls._extract_value(cleaned_output, "action")
            _action_input = cls._extract_value(cleaned_output, "action_input")
            return action_format.format(_action=_action, _action_input=_action_input)
        except Exception:
            if cleaned_output.startswith("Action:"):
                lines = cleaned_output.splitlines()
                _action = lines[1].strip()
                _action_input = textwrap.dedent("\n".join(lines[2:])).strip()
                return action_format.format(_action=_action, _action_input=_action_input)
            else:
                _action_input = cleaned_output

            return action_format.format(_action="Final Answer", _action_input=_action_input)

    @classmethod
    def extract_tool_response_for_llm(cls, text, tool_style: str = "code", max_token: int = 250) -> str:
        wrap_format = TOOL_RESPONSE_FORMAT
        tool_observation_format = TOOL_FORMAT[tool_style]
        cleaned_output = text.strip()
        if tool_style == "plugin":
            max_token = None

        try:
            _result = cls.truncate_text(cls._extract_value(cleaned_output, "result"), max_token)
            _intermediate_steps = cls.truncate_text(
                cls._extract_value(cleaned_output, "intermediate_steps"), max_token
            )
            _intermediate_steps = _intermediate_steps.replace("\\n", "\n").strip("\n")
            _result = _result.replace("\\n", "\n").strip("\n")
            _response = tool_observation_format.format(_intermediate_steps=_intermediate_steps, _result=_result)

            return wrap_format.format(_response=_response)
        except:
            if cleaned_output.startswith("Final Answer:"):
                lines = cleaned_output.splitlines()
                _response = textwrap.dedent("\n".join(lines[2:])).strip()
                _response = cls.truncate_text(_response, max_token)
                return wrap_format.format(_response=_response)

            _response = cls.truncate_text(cleaned_output, max_token)
            return wrap_format.format(_response=_response)

    @classmethod
    def extract_code_for_python_tool(cls, text: str, max_token: int = 2500, trunc_ratio: float = 0.2) -> str:
        whole_code = MessageDataModel._extract_response(text)
        trunc_code = cls.truncate_text(whole_code, max_token=max_token, trunc_ratio=trunc_ratio)
        return trunc_code

    @classmethod
    def extract_code_for_sql_tool(cls, text: str, max_token: int = 2500, trunc_ratio: float = 0.2) -> str:
        whole_code = MessageDataModel._extract_response(text)
        trunc_code = cls.truncate_text(whole_code, max_token=max_token, trunc_ratio=trunc_ratio)
        return trunc_code
def indent_multiline_string(multiline_string: str, indent: int = 1) -> str:
    return "\n".join("\t" * indent + line for line in multiline_string.split("\n"))
import subprocess
import sys
from copy import deepcopy
from typing import Any, Dict, Union

import pandas as pd
from sqlalchemy import create_engine
import tiktoken

from real_agents.adapters.schema import SQLDatabase


def convert(
    table_data: Union[pd.DataFrame, Dict[str, Any]], table_name: str = "table", visible_rows_num: int = 3
) -> Dict[str, str]:
    """
    Convert table data to string representations in different formats.

    :param table_data: A dictionary with "cols" (list of strings) and "rows"
                        (list of lists of strings) as keys.
    :param table_name: The name of the table.
    :param visible_rows_num: The number of rows to be displayed in the representation.
    :return: A dictionary with the string table representations in different formats.
    """

    def install_required_packages() -> None:
        packages = ["tabulate", "prettytable"]

        for package in packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Call the function to install the required packages
    install_required_packages()
    from prettytable import PrettyTable

    # Handle situation when the table_data is already a dataframe, FIXME: this is a hack
    new_table_data = {}
    if isinstance(table_data, pd.DataFrame):
        new_table_data["cols"] = table_data.columns
        new_table_data["rows"] = table_data.values.tolist()
    table_data = new_table_data

    # Type check for table_data
    if not isinstance(table_data, dict) or "cols" not in table_data or "rows" not in table_data:
        raise TypeError("table_data must be a dictionary with 'cols' and 'rows' as keys.")

    table_data_for_observable = deepcopy(table_data)
    if len(table_data_for_observable["rows"]) > visible_rows_num:
        table_data_for_observable["rows"] = table_data_for_observable["rows"][:visible_rows_num]
        table_data_for_observable["rows"].append(["..."] * len(table_data_for_observable["cols"]))

    # Create dataframe from table_data
    df = pd.DataFrame(table_data_for_observable["rows"], columns=table_data_for_observable["cols"])

    # Generate tables in different formats
    markdown_table = df.to_markdown(index=False)
    html_table = df.to_html(index=False)
    latex_table = df.to_latex(index=False)
    csv_table = df.to_csv(index=False)
    tsv_table = df.to_csv(index=False, sep="\t")
    rest_table = df.to_string(index=False)

    def bbcode_mode_table(data_frame: pd.DataFrame) -> str:
        bbcode_table = "[table]\n"
        for row in data_frame.itertuples(index=False):
            bbcode_table += "[tr]\n"
            for value in row:
                bbcode_table += f"[td]{value}[/td]\n"
            bbcode_table += "[/tr]\n"
        bbcode_table += "[/table]"
        return bbcode_table

    def mediawiki_mode_table(data_frame: pd.DataFrame) -> str:
        mediawiki_table = '{| class="wikitable"\n|-\n'
        for col in data_frame.columns:
            mediawiki_table += f"! {col}\n"
        for row in data_frame.itertuples(index=False):
            mediawiki_table += "|-\n"
            for value in row:
                mediawiki_table += f"| {value}\n"
        mediawiki_table += "|}"
        return mediawiki_table

    def org_mode_table(data_frame: pd.DataFrame) -> str:
        org_table = (
            "| "
            + " | ".join(data_frame.columns)
            + " |\n|-"
            + " | -".join(["-" * len(col) for col in data_frame.columns])
            + " |\n"
        )
        for row in data_frame.itertuples(index=False):
            org_table += "| " + " | ".join([str(value) for value in row]) + " |\n"
        return org_table

    bbcode_table = bbcode_mode_table(df)
    mediawiki_table = mediawiki_mode_table(df)
    org_table = org_mode_table(df)

    pretty_table = PrettyTable()
    pretty_table.field_names = table_data["cols"]
    for row in table_data["rows"]:
        pretty_table.add_row(row)
    pretty_table = str(pretty_table)

    # New function to generate SQL table
    def sql_mode_table(data_frame: pd.DataFrame, _table_name: str) -> str:
        sql_table_str = f"CREATE TABLE {table_name}(\n"

        for col in data_frame.columns:
            sql_table_str += f"{col} text,\n"

        # Remove the last comma and add the primary key constraint
        sql_table_str = sql_table_str[:-2] + f",\nPRIMARY KEY ({data_frame.columns[0]})\n);"

        sql_table_str += "\n/*\n{} example rows:\n".format(len(data_frame))
        for i, _row in data_frame.iterrows():
            _row = "\t".join([str(_cell) for _cell in _row.to_list()])
            sql_table_str += f"{_row}\n"
        sql_table_str += "*/"

        return sql_table_str

    sql_table = sql_mode_table(df, table_name)

    # Return the representation in different formats as a dictionary
    return {
        "Markdown": markdown_table,
        "HTML": html_table,
        "LaTeX": latex_table,
        "CSV": csv_table,
        "TSV": tsv_table,
        "reStructuredText": rest_table,
        "BBCode": bbcode_table,
        "MediaWiki": mediawiki_table,
        "Org mode": org_table,
        "PrettyTable": pretty_table,
        "SQL": sql_table,
    }


def serialize_df(
    table_data: pd.DataFrame,
    table_name: str,
    table_path: str,
    serialize_method: str = "tsv",
    num_visible_rows: int = 3,
    max_tokens: int = 1000,
    data_dir_splitter: str = "backend/data/",
) -> str:
    """Convert dataframe to a string representation."""
    if serialize_method == "tsv":
        # Here it means ignore the "path/to/the/data/<user_id/" part of the path
        pretty_path = "/".join(table_path.split(data_dir_splitter)[-1].strip("/").split("/")[1:])
        string = (
            "Here are table columns and the first {} rows of the table from the path {}"
            '(only a small part of the whole table) called "{}":\n'.format(num_visible_rows, pretty_path, table_name)
        )
        string += table_data.head(num_visible_rows).to_csv(sep="\t", index=False)
        # Truncate the string if it is too long
        enc = tiktoken.get_encoding("cl100k_base")
        enc_tokens = enc.encode(string)
        if len(enc_tokens) > max_tokens:
            string = enc.decode(enc_tokens[:max_tokens])
    elif serialize_method == "database":
        engine = create_engine("sqlite:///:memory:")
        table_data.to_sql(table_name, engine)
        db = SQLDatabase(engine)
        # TODO: Now access the internal variable
        setattr(db, "_sample_rows_in_table_info", num_visible_rows)
        string = db.get_table_info()
    else:
        raise ValueError("Unknown serialization method.")
    return string
import subprocess
import sys
from typing import Dict, List, Tuple


def convert(kg_input: List[Tuple], name_space: str = "") -> Dict[str, str]:
    """
    Convert knowledge graph data to string representations in different formats.

    :param kg_input: the list of knowledge graph triples.
    :param name_space: of the knowledge graph.
    :return: A dictionary with the string knowledge graph representations in different formats.
    """

    def install_required_packages() -> None:
        packages = ["rdflib", "rdflib-jsonld"]

        for package in packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Call the function to install the required packages
    install_required_packages()
    from rdflib import Graph, Namespace, URIRef

    g = Graph()

    # Define a namespace for the knowledge graph
    kg_ns = Namespace(name_space)
    g.bind("kg", kg_ns)

    # Add the triples to the graph
    for s, p, o in kg_input:
        subject = URIRef(kg_ns[s])
        predicate = URIRef(kg_ns[p])
        object = URIRef(kg_ns[o])
        g.add((subject, predicate, object))

    # Serialize the graph into the desired format
    representations = {_format: g.serialize(format=_format) for _format in ["json-ld", "turtle", "n3", "nt"]}

    return representations
import sqlite3
from typing import Dict, Union

import pandas as pd
import tiktoken


from real_agents.adapters.data_model.templates.skg_templates.table_templates import (
    convert as convert_table,
)
from real_agents.adapters.schema import SQLDatabase


def convert(db_input: Union[str, Dict[str, pd.DataFrame]], visible_rows_num: int = 3) -> Dict[str, str]:
    """
    Convert database data to string representations in different formats.

    :param db_input: the path to the sqlite database file, or a pd.DataFrame.
    :param visible_rows_num: the number of rows to be displayed in each table.
    :return: A dictionary with the string database representations in different formats.
    """
    if isinstance(db_input, str):
        conn = sqlite3.connect(db_input)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [name[0] for name in cursor.fetchall()]
        dfs = {table_name: pd.read_sql_query(f"SELECT * FROM {table_name}", conn) for table_name in table_names}
    elif isinstance(db_input, dict) and all(isinstance(df, pd.DataFrame) for df in db_input.values()):
        dfs = db_input
    else:
        raise ValueError("db_input should be either a SQLite database file path or a dictionary of pandas DataFrames")

    representations = {
        "Markdown": "",
        "HTML": "",
        "LaTeX": "",
        "CSV": "",
        "TSV": "",
        "reStructuredText": "",
        "BBCode": "",
        "MediaWiki": "",
        "Org mode": "",
        "PrettyTable": "",
        "SQL": "",
    }

    for table_name, df in dfs.items():
        table_data = {"cols": df.columns.tolist(), "rows": df.values.tolist()}
        table_representations = convert_table(table_data, table_name, visible_rows_num)
        for _format, table_representation in table_representations.items():
            representations[_format] += table_representation + "\n\n"

    return representations


def serialize_db(
    db: SQLDatabase,
    serialize_method: str = "database",
    num_visible_rows: int = 3,
    max_tokens: int = 1000,
) -> str:
    """Convert database engine to a string representation."""
    if serialize_method == "database":
        # TODO: Now access the internal variable
        setattr(db, "_sample_rows_in_table_info", num_visible_rows)
        string = db.get_table_info()
        # Truncate the string if it is too long
        enc = tiktoken.get_encoding("cl100k_base")
        enc_tokens = enc.encode(string)
        if len(enc_tokens) > max_tokens:
            string = enc.decode(enc_tokens[:max_tokens])
    else:
        raise ValueError("Unknown serialization method.")
    return string
import json
from copy import deepcopy
from typing import Dict, List

from real_agents.adapters.data_model.base import DataModel


class JsonDataModel(DataModel):
    """A data model for json, general purpose."""

    filter_keys: List[str] = []

    def get_llm_side_data(self, json_format: str = "json") -> str:
        if json_format == "json":
            assert isinstance(self.raw_data, Dict)
            llm_side_data = deepcopy(self.raw_data)
            for key, value in self.raw_data.items():
                if key in self.filter_keys:
                    llm_side_data[key] = "..."
                    continue

                if isinstance(value, DataModel):
                    llm_side_data[key] = value.get_llm_side_data()
                else:
                    llm_side_data[key] = str(value)

            return json.dumps(llm_side_data, indent=4)
        else:
            raise NotImplementedError
from real_agents.adapters.data_model.base import DataModel


class TextDataModel(DataModel):
    """A data model for text, general purpose."""

    def get_llm_side_data(self, max_chars: int = 5000) -> str:
        assert isinstance(self.raw_data, str)
        return self.raw_data[:max_chars]
from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel


class DataModel(BaseModel):
    """Base class for data models."""

    id: str
    raw_data: Any
    raw_data_name: str
    raw_data_path: str
    llm_side_data: Any  # could be string or potentially images for future needs
    human_side_data: Any

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_raw_data(
        cls, raw_data: Any, raw_data_name: str = "<default_name>", raw_data_path: str = "<default_path>", **kwargs: Any
    ) -> DataModel:
        uid = str(uuid.uuid4())
        return cls(id=uid, raw_data=raw_data, raw_data_name=raw_data_name, raw_data_path=raw_data_path, **kwargs)

    def get_id(self) -> str:
        return self.id

    def get_raw_data(self) -> Any:
        return self.raw_data

    def get_llm_side_data(self) -> Any:
        return self.raw_data

    def get_human_side_data(self) -> Any:
        return self.raw_data

    def __str__(self) -> str:
        return self.get_llm_side_data()
from __future__ import annotations

import json
from typing import Any

from pandas import DataFrame

from real_agents.adapters.data_model.base import DataModel
from real_agents.adapters.data_model.templates.skg_templates.table_templates import serialize_df


class TableDataModel(DataModel):
    """A data model for table."""

    db_view: DataModel = None

    def set_db_view(self, db_data_model: DataModel) -> None:
        self.db_view = db_data_model

    def get_llm_side_data(self, serialize_method: str = "tsv", num_visible_rows: int = 3) -> Any:
        # Show the first few rows for observation.
        table_data = self.raw_data
        table_name = self.raw_data_name
        table_path = self.raw_data_path
        formatted_table = serialize_df(table_data, table_name, table_path, serialize_method, num_visible_rows)
        return formatted_table

    def get_human_side_data(self, mode: str = "HEAD") -> Any:
        # We support different mode for the front-end display.
        # For `HEAD` mode, we show the first few rows for observation.
        if mode == "HEAD":
            return self.raw_data.head()
        elif mode == "FULL":
            return self.raw_data
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def to_react_table(table: DataFrame) -> str:
        columns = list(map(lambda item: {"accessorKey": item, "header": item}, table.columns.tolist()))
        # FIXME: NaN may not be handled here.
        data = table.fillna("").to_dict(orient="records")
        table = json.dumps({"columns": columns, "data": data})
        return table
from __future__ import annotations

import os
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect

from real_agents.adapters.data_model.base import DataModel
from real_agents.adapters.data_model.table import TableDataModel
from real_agents.adapters.data_model.templates.skg_templates.database_templates import serialize_db
from real_agents.adapters.schema import SQLDatabase


class DatabaseDataModel(DataModel):
    """A data model for database."""

    @classmethod
    def from_table_data_model(cls, table_data_model: TableDataModel) -> DatabaseDataModel:
        os.makedirs(f".db_cache/{table_data_model.id}", exist_ok=True)
        db_path = os.path.join(
            f".db_cache/{table_data_model.id}", os.path.splitext(table_data_model.raw_data_name)[0] + ".db"
        )
        engine = create_engine(f"sqlite:///{db_path}")
        table_data_model.raw_data.to_sql(table_data_model.raw_data_name, engine, if_exists="replace")
        db = SQLDatabase(engine)
        return cls.from_raw_data(raw_data=db, raw_data_name=table_data_model.raw_data_name)

    def insert_table_data_model(self, table_data_model: TableDataModel) -> None:
        engine = self.raw_data.engine
        table_data_model.raw_data.to_sql(table_data_model.raw_data_name, engine)

    def get_llm_side_data(self, serialize_method: str = "database", num_visible_rows: int = 3) -> Any:
        db = self.raw_data
        formatted_db = serialize_db(db, serialize_method, num_visible_rows)
        return formatted_db

    def get_human_side_data(self) -> Any:
        # In the frontend, we show the first few rows of each table
        engine = self.raw_data.engine
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        # Loop through each table name, creating a DataFrame from the first three rows of each table
        df_dict = {}
        for table_name in table_names:
            query = f"SELECT * FROM {table_name} LIMIT 3"
            df = pd.read_sql(query, engine)
            df_dict[table_name] = df
        return df_dict
from real_agents.adapters.data_model.base import DataModel
from real_agents.adapters.data_model.database import DatabaseDataModel
from real_agents.adapters.data_model.image import ImageDataModel
from real_agents.adapters.data_model.json import JsonDataModel
from real_agents.adapters.data_model.message import MessageDataModel
from real_agents.adapters.data_model.kaggle import KaggleDataModel
from real_agents.adapters.data_model.plugin import APIYamlModel, SpecModel
from real_agents.adapters.data_model.table import TableDataModel

__all__ = [
    "DataModel",
    "TableDataModel",
    "DatabaseDataModel",
    "ImageDataModel",
    "JsonDataModel",
    "KaggleDataModel",
    "APIYamlModel",
    "SpecModel",
    "MessageDataModel",
    "HTMLDataModel",
]
from __future__ import annotations

from typing import Any, Optional, Sequence

from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool

from real_agents.adapters.agent_helpers import AgentExecutor
from real_agents.data_agent.copilot import ConversationalChatAgent
from real_agents.plugins_agent.plugin import ConversationalPluginChatAgent
from real_agents.web_agent.webot import ConversationalWebotChatAgent


def initialize_agent(
    tools: Sequence[BaseTool],
    llm: BaseLanguageModel,
    continue_model: str = None,
    agent_kwargs: Optional[dict] = None,
    return_intermediate_steps: Optional[bool] = True,
    **kwargs: Any,
) -> AgentExecutor:
    """Load an agent executor given tools and LLM.

    Args:
        tools: List of tools this agent has access to.
        llm: Language model to use as the agent.
        callback_manager: CallbackManager to use. Global callback manager is used if
            not provided. Defaults to None.
        agent_kwargs: Additional key word arguments to pass to the underlying agent_executor
        return_intermediate_steps: Whether to return intermediate steps in the agent
        **kwargs: Additional key word arguments passed to the agent executor

    Returns:
        An agent executor
    """

    agent_kwargs = agent_kwargs or {}
    agent_obj = ConversationalChatAgent.from_llm_and_tools(
        llm=llm, tools=tools, continue_model=continue_model, **agent_kwargs
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent_obj,
        tools=tools,
        return_intermediate_steps=return_intermediate_steps,
        **kwargs,
    )
    return agent_executor


def initialize_plugin_agent(
    tools: Sequence[BaseTool],
    llm: BaseLanguageModel,
    continue_model: str = None,
    agent_kwargs: Optional[dict] = None,
    return_intermediate_steps: Optional[bool] = True,
    **kwargs: Any,
) -> AgentExecutor:
    """Load an agent executor given tools and LLM.

    Args:
        tools: List of tools this agent has access to.
        llm: Language model to use as the agent.
        agent_kwargs: Additional key word arguments to pass to the underlying agent_executor
        return_intermediate_steps: Whether to return intermediate steps in the agent
        **kwargs: Additional key word arguments passed to the agent executor

    Returns:
        An agent executor
    """

    agent_kwargs = agent_kwargs or {}
    agent_obj = ConversationalPluginChatAgent.from_llm_and_tools(
        llm=llm, tools=tools, continue_model=continue_model, **agent_kwargs
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent_obj,
        tools=tools,
        return_intermediate_steps=return_intermediate_steps,
        **kwargs,
    )
    return agent_executor


def initialize_webot_agent(
    tools: Sequence[BaseTool],
    llm: BaseLanguageModel,
    continue_model: str = None,
    agent_kwargs: Optional[dict] = None,
    return_intermediate_steps: Optional[bool] = True,
    **kwargs: Any,
) -> AgentExecutor:
    """Load an agent executor given tools and LLM.

    Args:
        tools: List of tools this agent has access to.
        llm: Language model to use as the agent.
        agent_kwargs: Additional key word arguments to pass to the underlying agent_executor
        return_intermediate_steps: Whether to return intermediate steps in the agent
        **kwargs: Additional key word arguments passed to the agent executor

    Returns:
        An agent executor
    """

    agent_kwargs = agent_kwargs or {}
    agent_obj = ConversationalWebotChatAgent.from_llm_and_tools(
        llm=llm, tools=tools, continue_model=continue_model, **agent_kwargs
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent_obj,
        tools=tools,
        return_intermediate_steps=return_intermediate_steps,
        **kwargs,
    )
    return agent_executor
from typing import NamedTuple
from langchain import SQLDatabase
from sqlalchemy import text
from sqlalchemy.engine import Row
from tabulate import tabulate
from typing import List, Any


class AgentTransition(NamedTuple):
    """Agent's transition to take."""

    return_values: dict
    log: str


EMPTY_RESULT_STR = "NONE"  # to show NONE result in front-end.


class SQLDatabase(SQLDatabase):
    @staticmethod
    def _pretty_format(headers: Any, result: List[Row]) -> str:
        dicts = [dict(zip(headers, row)) for row in result]
        tab_result = tabulate(tabular_data=dicts, headers="keys", tablefmt="psql")

        if tab_result == "":
            return EMPTY_RESULT_STR

        return tab_result

    def run(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self._engine.begin() as connection:
            if self._schema is not None:
                connection.exec_driver_sql(f"SET search_path TO {self._schema}")
            cursor = connection.execute(text(command))
            if cursor.returns_rows:
                headers = cursor.keys()
                if fetch == "all":
                    result = cursor.fetchall()
                elif fetch == "one":
                    # result = cursor.fetchone()[0]  # type: ignore
                    result = [cursor.fetchone()]  # type: ignore
                else:
                    raise ValueError("Fetch parameter must be either 'one' or 'all'")

                # pretty format
                tab_result = self._pretty_format(headers, result)
                return tab_result
        return ""
"""Interface for tools."""
from inspect import signature
from typing import Any, Awaitable, Callable, Dict, Optional, Type, Union
from pydantic import BaseModel, validate_arguments

from langchain.tools.base import BaseTool

from real_agents.adapters.data_model import DataModel
from real_agents.adapters.callbacks.manager import (
    CallbackManager,
    Callbacks,
)


class Tool(BaseTool):
    """Tool that takes in function or coroutine directly."""

    description: str = ""
    func: Callable[..., str]
    """The function to run when the tool is called."""
    coroutine: Optional[Callable[..., Awaitable[str]]] = None
    """The asynchronous version of the function."""

    @property
    def args(self) -> dict:
        if self.args_schema is not None:
            return self.args_schema.schema()["properties"]
        else:
            inferred_model = validate_arguments(self.func).model  # type: ignore
            schema = inferred_model.schema()["properties"]
            valid_keys = signature(self.func).parameters
            return {k: schema[k] for k in valid_keys}

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Use the tool."""
        return self.func(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Use the tool asynchronously."""
        if self.coroutine:
            return await self.coroutine(*args, **kwargs)
        raise NotImplementedError("Tool does not support async")

    # TODO: this is for backwards compatibility, remove in future
    def __init__(
        self, name: str, func: Callable[[str], Union[Dict[Any, Any], DataModel]], description: str, **kwargs: Any
    ) -> None:
        """Initialize tool."""
        super(Tool, self).__init__(name=name, func=func, description=description, **kwargs)

    def run(
        self,
        tool_input: Union[str, Dict],
        verbose: Optional[bool] = None,
        start_color: Optional[str] = "green",
        color: Optional[str] = "green",
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Any:
        """Run the tool."""
        parsed_input = self._parse_input(tool_input)
        if not self.verbose and verbose is not None:
            verbose_ = verbose
        else:
            verbose_ = self.verbose

        # todo: fix this place
        callback_manager = CallbackManager.configure(callbacks, self.callbacks, verbose=verbose_)
        # TODO: maybe also pass through run_manager is _run supports kwargs
        new_arg_supported = signature(self._run).parameters.get("run_manager")
        run_manager = callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input if isinstance(tool_input, str) else str(tool_input),
            color=start_color,
            **kwargs,
        )
        try:
            tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)
            observation = (
                self._run(*tool_args, run_manager=run_manager, **tool_kwargs)
                if new_arg_supported
                else self._run(*tool_args, **tool_kwargs)
            )
        except (Exception, KeyboardInterrupt) as e:
            run_manager.on_tool_error(e)
            raise e

        run_manager.on_tool_end(observation, color=color, name=self.name, **kwargs)

        return observation


class InvalidTool(BaseTool):
    """Tool that is run when invalid tool name is encountered by agent."""

    name = "invalid_tool"
    description = "Called when tool name is invalid."

    def _run(self, tool_name: str) -> str:
        """Use the tool."""
        return f"{tool_name} is not a valid tool, try another one."

    async def _arun(self, tool_name: str) -> str:
        """Use the tool asynchronously."""
        return f"{tool_name} is not a valid tool, try another one."


def tool(
    *args: Union[str, Callable],
    return_direct: bool = False,
    args_schema: Optional[Type[BaseModel]] = None,
    infer_schema: bool = True,
) -> Callable:
    """Make tools out of functions, can be used with or without arguments.

    Args:
        *args: The arguments to the tool.
        return_direct: Whether to return directly from the tool rather
            than continuing the agent loop.
        args_schema: optional argument schema for user to specify
        infer_schema: Whether to infer the schema of the arguments from
            the function's signature. This also makes the resultant tool
            accept a dictionary input to its `run()` function.

    Requires:
        - Function must be of type (str) -> str
        - Function must have a docstring

    Examples:
        .. code-block:: python

            @tool
            def search_api(query: str) -> str:
                # Searches the API for the query.
                return

            @tool("search", return_direct=True)
            def search_api(query: str) -> str:
                # Searches the API for the query.
                return
    """

    def _make_with_name(tool_name: str) -> Callable:
        def _make_tool(func: Callable) -> Tool:
            assert func.__doc__, "Function must have a docstring"
            # Description example:
            # search_api(query: str) - Searches the API for the query.
            description = f"{tool_name}{signature(func)} - {func.__doc__.strip()}"
            _args_schema = args_schema
            if _args_schema is None and infer_schema:
                _args_schema = validate_arguments(func).model  # type: ignore
            tool_ = Tool(
                name=tool_name,
                func=func,
                args_schema=_args_schema,
                description=description,
                return_direct=return_direct,
            )
            return tool_

        return _make_tool

    if len(args) == 1 and isinstance(args[0], str):
        # if the argument is a string, then we use the string as the tool name
        # Example usage: @tool("search", return_direct=True)
        return _make_with_name(args[0])
    elif len(args) == 1 and callable(args[0]):
        # if the argument is a function, then we use the function name as the tool name
        # Example usage: @tool
        return _make_with_name(args[0].__name__)(args[0])
    elif len(args) == 0:
        # if there are no arguments, then we use the function name as the tool name
        # Example usage: @tool(return_direct=True)
        def _partial(func: Callable[[str], str]) -> BaseTool:
            return _make_with_name(func.__name__)(func)

        return _partial
    else:
        raise ValueError("Too many arguments for tool decorator")
"""Chain that takes in an input and produces an action and action input."""
from __future__ import annotations

import json
import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import yaml
from pydantic import BaseModel, root_validator

from langchain.agents.agent_types import AgentType
from langchain.agents.tools import InvalidTool
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)
from langchain.chains.base import Chain
from langchain.input import get_color_mapping
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    BaseOutputParser,
    OutputParserException,
)
from langchain.tools.base import BaseTool

from real_agents.adapters.llm import LLMChain
from real_agents.adapters.data_model import DataModel, MessageDataModel

logger = logging.getLogger(__name__)


class BaseSingleActionAgent(BaseModel):
    """Base Agent class."""

    @property
    def return_values(self) -> List[str]:
        """Return values of the agent."""
        return ["output"]

    def get_allowed_tools(self) -> Optional[List[str]]:
        return None

    @abstractmethod
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """

    @abstractmethod
    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations."""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish({"output": "Agent stopped due to iteration limit or time limit."}, "")
        else:
            raise ValueError(f"Got unsupported early_stopping_method `{early_stopping_method}`")

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> BaseSingleActionAgent:
        raise NotImplementedError

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of agent."""
        _dict = super().dict()
        _type = self._agent_type
        if isinstance(_type, AgentType):
            _dict["_type"] = str(_type.value)
        else:
            _dict["_type"] = _type
        return _dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the agent.

        Args:
            file_path: Path to file to save the agent to.

        Example:
        .. code-block:: python

            # If working with agent executor
            agent.agent.save(file_path="path/agent.yaml")
        """
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        agent_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")

    def tool_run_logging_kwargs(self) -> Dict:
        return {}


class AgentOutputParser(BaseOutputParser):
    @abstractmethod
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse text into agent action/finish."""


class Agent(BaseSingleActionAgent):
    """Class responsible for calling the language model and deciding the action.

    This is driven by an LLMChain. The prompt in the LLMChain MUST include
    a variable called "agent_scratchpad" where the agent can put its
    intermediary work.
    """

    llm_chain: LLMChain
    output_parser: AgentOutputParser
    allowed_tools: Optional[List[str]] = None

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of agent."""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def get_allowed_tools(self) -> Optional[List[str]]:
        return self.allowed_tools

    @property
    def return_values(self) -> List[str]:
        return ["output"]

    def _fix_text(self, text: str) -> str:
        """Fix the text."""
        raise ValueError("fix_text not implemented for this agent.")

    @property
    def _stop(self) -> List[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, full_observation in intermediate_steps:
            thoughts += action.log
            observation = (
                full_observation.get_llm_side_data() if isinstance(full_observation, DataModel) else full_observation
            )
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        return thoughts

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_inputs = MessageDataModel.truncate_chat_history(full_inputs)
        full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)
        return self.output_parser.parse(full_output)

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = await self.llm_chain.apredict(callbacks=callbacks, **full_inputs)
        return self.output_parser.parse(full_output)

    def get_full_inputs(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts = self._construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        full_inputs = {**kwargs, **new_inputs}
        return full_inputs

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return list(set(self.llm_chain.input_keys) - {"agent_scratchpad"})

    @root_validator()
    def validate_prompt(cls, values: Dict) -> Dict:
        """Validate that prompt matches format."""
        prompt = values["llm_chain"].prompt
        if "agent_scratchpad" not in prompt.input_variables:
            logger.warning(
                "`agent_scratchpad` should be a variable in prompt.input_variables."
                " Did not find it, so adding it at the end."
            )
            prompt.input_variables.append("agent_scratchpad")
            if isinstance(prompt, PromptTemplate):
                prompt.template += "\n{agent_scratchpad}"
            elif isinstance(prompt, FewShotPromptTemplate):
                prompt.suffix += "\n{agent_scratchpad}"
            else:
                raise ValueError(f"Got unexpected prompt type {type(prompt)}")
        return values

    @property
    @abstractmethod
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""

    @property
    @abstractmethod
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""

    @classmethod
    @abstractmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Create a prompt for this class."""

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        """Validate that appropriate tools are passed in."""
        pass

    @classmethod
    @abstractmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        """Get default output parser for this class."""

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        llm_chain = LLMChain(
            llm=llm,
            prompt=cls.create_prompt(tools),
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser()
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """Return response when agent has been stopped due to max iterations."""
        if early_stopping_method == "force":
            # `force` just returns a constant string
            return AgentFinish({"output": "Agent stopped due to iteration limit or time limit."}, "")
        elif early_stopping_method == "generate":
            # Generate does one final forward pass
            thoughts = ""
            for action, full_observation in intermediate_steps:
                thoughts += action.log
                observation = (
                    full_observation.get_llm_side_data()
                    if isinstance(full_observation, DataModel)
                    else full_observation
                )
                thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
            # Adding to the previous steps, we now tell the LLM to make a final pred
            thoughts += "\n\nI now need to return a final answer based on the previous steps:"
            new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
            full_inputs = {**kwargs, **new_inputs}
            full_output = self.llm_chain.predict(**full_inputs)
            # We try to extract a final answer
            parsed_output = self.output_parser.parse(full_output)
            if isinstance(parsed_output, AgentFinish):
                # If we can extract, we send the correct stuff
                return parsed_output
            else:
                # If we can extract, but the tool is not the final tool,
                # we just return the full output
                return AgentFinish({"output": full_output}, full_output)
        else:
            raise ValueError(
                "early_stopping_method should be one of `force` or `generate`, " f"got {early_stopping_method}"
            )

    def tool_run_logging_kwargs(self) -> Dict:
        return {
            "llm_prefix": self.llm_prefix,
            "observation_prefix": self.observation_prefix,
        }


class ExceptionTool(BaseTool):
    name = "_Exception"
    description = "Exception tool"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return query

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return query


class AgentExecutor(Chain):
    """Consists of an agent using tools."""

    agent: BaseSingleActionAgent
    tools: Sequence[BaseTool]
    return_intermediate_steps: bool = False
    max_iterations: Optional[int] = 5
    max_execution_time: Optional[float] = None
    early_stopping_method: str = "force"
    handle_parsing_errors: Union[bool, str, Callable[[OutputParserException], str]] = False

    @classmethod
    def from_agent_and_tools(
        cls,
        agent: BaseSingleActionAgent,
        tools: Sequence[BaseTool],
        **kwargs: Any,
    ) -> AgentExecutor:
        """Create from agent and tools."""
        return cls(agent=agent, tools=tools, **kwargs)

    @root_validator()
    def validate_tools(cls, values: Dict) -> Dict:
        """Validate that tools are compatible with agent."""
        agent = values["agent"]
        tools = values["tools"]
        allowed_tools = agent.get_allowed_tools()
        if allowed_tools is not None:
            if set(allowed_tools) != set([tool.name for tool in tools]):
                raise ValueError(
                    f"Allowed tools ({allowed_tools}) different than "
                    f"provided tools ({[tool.name for tool in tools]})"
                )
        return values

    @root_validator()
    def validate_return_direct_tool(cls, values: Dict) -> Dict:
        """Validate that tools are compatible with agent."""
        return values

    def save(self, file_path: Union[Path, str]) -> None:
        """Raise error - saving not supported for Agent Executors."""
        raise ValueError(
            "Saving not supported for agent executors. "
            "If you are trying to save the agent, please use the "
            "`.save_agent(...)`"
        )

    def save_agent(self, file_path: Union[Path, str]) -> None:
        """Save the underlying agent."""
        return self.agent.save(file_path)

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return self.agent.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if self.return_intermediate_steps:
            return self.agent.return_values + ["intermediate_steps"]
        else:
            return self.agent.return_values

    def lookup_tool(self, name: str) -> BaseTool:
        """Lookup tool by name."""
        return {tool.name: tool for tool in self.tools}[name]

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        if self.max_execution_time is not None and time_elapsed >= self.max_execution_time:
            return False

        return True

    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidTool().run(
                    agent_action.tool,
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            result.append((agent_action, observation))
        return result

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping([tool.name for tool in self.tools], excluded_colors=["green"])
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(next_step_output, intermediate_steps, run_manager=run_manager)

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(tool_return, intermediate_steps, run_manager=run_manager)
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(self.early_stopping_method, intermediate_steps, **inputs)
        return self._return(output, intermediate_steps, run_manager=run_manager)

    def _get_tool_return(self, next_step_output: Tuple[AgentAction, Union[str, DataModel]]) -> Optional[AgentFinish]:
        """Check if the tool is a returning tool."""
        agent_action, full_observation = next_step_output
        observation = full_observation
        if isinstance(full_observation, DataModel):
            llm_raw_observation = full_observation.get_llm_side_data()
            observation = MessageDataModel.extract_tool_response_for_llm(
                llm_raw_observation, tool_style=self.memory.style
            )
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # Invalid tools won't be in the map, so we return False.
        if agent_action.tool in name_to_tool_map:
            if name_to_tool_map[agent_action.tool].return_direct:
                return AgentFinish(
                    {self.agent.return_values[0]: observation},
                    "",
                )
        return None
from __future__ import annotations

from typing import Optional, Union
from pydantic import Extra

from langchain.schema import (
    AgentAction,
    AgentFinish,
)
from real_agents.adapters.agent_helpers.agent import AgentOutputParser
from real_agents.adapters.schema import AgentTransition


class ConversationOutputParser(AgentOutputParser):
    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow
        arbitrary_types_allowed = True

    def get_format_instructions(self, app_name="copilot") -> str:
        from real_agents.data_agent.copilot_prompt import FORMAT_INSTRUCTIONS as COPILOT_FORMAT_INSTRUCTIONS
        from real_agents.plugins_agent.plugin_prompt import FORMAT_INSTRUCTIONS as PLUGINS_FORMAT_INSTRUCTIONS
        from real_agents.web_agent.webot_prompt import FORMAT_INSTRUCTIONS as WEBOT_FORMAT_INSTRUCTIONS

        if app_name == "copilot":
            return COPILOT_FORMAT_INSTRUCTIONS
        elif app_name == "webot":
            return WEBOT_FORMAT_INSTRUCTIONS
        elif app_name == "plugins":
            return PLUGINS_FORMAT_INSTRUCTIONS
        else:
            raise ValueError(f"Unknown app_name {app_name}")

    def parse(self, text: str) -> Union[AgentTransition, AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        import re

        def _extract_explanation(json_string: str) -> Optional[str]:
            if "```" in json_string:
                return json_string.split("```")[0]
            else:
                return None

        def _extract_value(json_string: str, key: str) -> str:
            pattern = re.compile(rf'"?{key}"?\s*:\s*("((?:[^"\\]|\\.)*)"|(\b[^,\s]*\b))', re.MULTILINE)
            match = pattern.search(json_string)
            if match:
                return match.group(1).replace('\\"', '"').replace("\\\\", "\\").strip('"').strip("'")

            raise ValueError(f"Could not find {key} in {json_string}")

        try:
            _action = _extract_value(cleaned_output, "action")
            _action_input = _extract_value(cleaned_output, "action_input")
            if _action == "Final Answer":
                return AgentFinish({"output": _action_input}, cleaned_output)

            # Transition sentence should only be used not final answer.
            _explanation = _extract_explanation(cleaned_output)
            return AgentAction(_action, _action_input, cleaned_output)
        except Exception:
            if cleaned_output.startswith("Action:"):
                lines = cleaned_output.splitlines()
                action = lines[1].strip()
                import textwrap

                action_input = textwrap.dedent("\n".join(lines[2:])).strip()
                return AgentAction(action, action_input, cleaned_output)

            return AgentFinish({"output": cleaned_output}, cleaned_output)

    @property
    def _type(self) -> str:
        return "conversational_chat"
"""Interface for agents."""
from real_agents.adapters.agent_helpers.agent import (
    Agent,
    AgentExecutor,
    AgentOutputParser,
    BaseSingleActionAgent,
)
from real_agents.adapters.agent_helpers.tools import Tool, tool

__all__ = [
    "AgentExecutor",
    "Agent",
    "Tool",
    "tool",
    "AgentOutputParser",
    "BaseSingleActionAgent",
]
from typing import Any, Dict, List, Optional, Tuple
from pydantic import root_validator

from langchain.memory.utils import get_prompt_input_key
from langchain.base_language import BaseLanguageModel
from langchain.schema import BaseMessage, get_buffer_string
from langchain.memory.chat_memory import BaseChatMemory, BaseMemory

from real_agents.adapters.data_model import DataModel, MessageDataModel


class ConversationBufferMemory(BaseChatMemory):
    """Buffer for storing conversation memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:

    @property
    def buffer(self) -> Any:
        """String buffer of memory."""
        if self.return_messages:
            return self.chat_memory.messages
        else:
            return get_buffer_string(
                self.chat_memory.messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}


class ConversationStringBufferMemory(BaseMemory):
    """Buffer for storing conversation memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    """Prefix to use for AI generated responses."""
    buffer: str = ""
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    memory_key: str = "history"  #: :meta private:

    @root_validator()
    def validate_chains(cls, values: Dict) -> Dict:
        """Validate that return messages is not True."""
        if values.get("return_messages", False):
            raise ValueError("return_messages must be False for ConversationStringBufferMemory")
        return values

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.
        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        human = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai = f"{self.ai_prefix}: " + outputs[output_key]
        self.buffer += "\n" + "\n".join([human, ai])

    def clear(self) -> None:
        """Clear memory contents."""
        self.buffer = ""


class ConversationReActBufferMemory(BaseChatMemory):
    """Buffer for storing conversational ReAct memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:
    max_token_limit: int = 2000
    llm: BaseLanguageModel = None
    style: str = "code"

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def action_prefix(self) -> str:
        """Prefix to append the action with."""
        return "Action:"

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    @property
    def llm_final(self) -> str:
        """Final Answer"""

    @property
    def buffer(self) -> List[BaseMessage]:
        """String buffer of memory."""
        if self.return_messages:
            return self.chat_memory.messages
        else:
            return get_buffer_string(
                self.chat_memory.messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}

    def _get_input_output(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Tuple[str, str]:
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) == 1:
                output_key = list(outputs.keys())[0]
                return inputs[prompt_input_key], outputs[output_key]
            else:
                assert "intermediate_steps" in outputs, "intermediate_steps must in outputs when output_key length > 1"
                intermediate_message = ""
                for action, full_observation in outputs["intermediate_steps"]:
                    intermediate_message += "\n{\n"
                    intermediate_message += (
                        '\t"action": "{}"'.format(action.tool) + "\n"
                    )  # todo: move to schema, as well as the one in prompt
                    intermediate_message += '\t"action_input": "{}"'.format(action.tool_input) + "\n"
                    intermediate_message += "}\n"
                    observation = full_observation
                    if isinstance(full_observation, DataModel):
                        llm_raw_observation = full_observation.get_llm_side_data()
                        observation = MessageDataModel.extract_tool_response_for_llm(
                            llm_raw_observation, tool_style=self.style
                        )
                    intermediate_message += "{}\n".format(observation)
                output = intermediate_message + outputs[list(outputs.keys())[0]]

                return inputs[prompt_input_key], output
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]

    def fit_max_token_limit(self):
        from real_agents.adapters.data_model import MessageDataModel

        # if self.llm != None:
        buffer = self.chat_memory.messages
        # curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        curr_buffer_length = MessageDataModel._count_tokens("\n".join([_.content for _ in buffer]))
        if curr_buffer_length > self.max_token_limit:
            while curr_buffer_length > self.max_token_limit:
                buffer.pop(0)
                curr_buffer_length = MessageDataModel._count_tokens("\n".join([_.content for _ in buffer]))
                # curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        self.chat_memory.messages = buffer

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer. Pruned."""
        super().save_context(inputs, outputs)
        self.fit_max_token_limit()
from typing import Any, Dict, List

from langchain.schema import BaseMemory


class ReadOnlySharedStringMemory(BaseMemory):
    """A memory wrapper that is read-only and cannot be changed."""

    memory: BaseMemory

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return self.memory.memory_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load memory variables from memory."""
        prev_memory_state = self.memory.return_messages
        self.memory.return_messages = False
        memory_string = self.memory.load_memory_variables(inputs)
        self.memory.return_messages = prev_memory_state
        return memory_string

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Nothing should be saved or changed"""
        pass

    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
        pass
from typing import Dict, List, Type

from langchain.schema import BaseMemory
from langchain.memory.chat_memory import BaseChatMemory

from real_agents.adapters.memory.buffer import (
    ConversationBufferMemory,
    ConversationStringBufferMemory,
)
from real_agents.adapters.memory.read_only_string_memory import ReadOnlySharedStringMemory
from real_agents.adapters.memory.buffer import ConversationReActBufferMemory



__all__ = [
    "ConversationBufferMemory",
    "ConversationReActBufferMemory",
    "ConversationStringBufferMemory",
    "BaseMemory",
    "BaseChatMemory",
    "ReadOnlySharedStringMemory",
]

type_to_cls_dict: Dict[str, Type[BaseMemory]] = {
    "chat_buffer": ConversationBufferMemory,
    "chat_string_buffer": ConversationStringBufferMemory,
}
from typing import Any, Dict, List, Optional
from pydantic import Extra

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.llms.anthropic import _AnthropicCommon
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)


class ChatAnthropic(BaseChatModel, _AnthropicCommon):
    r"""Wrapper around Anthropic's large language model.

    To use, you should have the ``anthropic`` python package installed, and the
    environment variable ``ANTHROPIC_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python
            import anthropic
            from langchain.llms import Anthropic
            model = ChatAnthropic(model="<model_name>", anthropic_api_key="my-api-key")
    """
    stop: Optional[List[str]] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anthropic-chat"

    def _convert_one_message_to_text(self, message: BaseMessage) -> str:
        if isinstance(message, ChatMessage):
            message_text = f"\n\n{message.role.capitalize()}: {message.content}"
        elif isinstance(message, HumanMessage):
            message_text = f"{self.HUMAN_PROMPT} {message.content}"
        elif isinstance(message, AIMessage):
            message_text = f"{self.AI_PROMPT} {message.content}"
        elif isinstance(message, SystemMessage):
            message_text = f"{self.HUMAN_PROMPT} <admin>{message.content}</admin>"
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _convert_messages_to_text(self, messages: List[BaseMessage]) -> str:
        """Format a list of strings into a single string with necessary newlines.

        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.

        Returns:
            str: Combined string with necessary newlines.
        """
        return "".join(self._convert_one_message_to_text(message) for message in messages)

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Format a list of messages into a full prompt for the Anthropic model

        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.

        Returns:
            str: Combined string with necessary HUMAN_PROMPT and AI_PROMPT tags.
        """
        if not self.AI_PROMPT:
            raise NameError("Please ensure the anthropic package is loaded")

        if not isinstance(messages[-1], AIMessage):
            messages.append(AIMessage(content=""))
        text = self._convert_messages_to_text(messages)
        return text.rstrip()  # trim off the trailing ' ' that might come from the "Assistant: "

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        prompt = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {"prompt": prompt, **self._default_params}
        if self.stop is not None:
            if stop is None:
                stop = self.stop
            else:
                stop.extend(self.stop)
        if stop:
            params["stop_sequences"] = stop

        if self.streaming:
            completion = ""
            stream_resp = self.client.completion_stream(**params)
            for data in stream_resp:
                delta = data["completion"][len(completion) :]
                completion = data["completion"]
                if run_manager:
                    run_manager.on_llm_new_token(
                        delta,
                    )
        else:
            response = self.client.completion(**params)
            completion = response["completion"]
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        prompt = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {"prompt": prompt, **self._default_params}
        if stop:
            params["stop_sequences"] = stop

        if self.streaming:
            completion = ""
            stream_resp = await self.client.acompletion_stream(**params)
            async for data in stream_resp:
                delta = data["completion"][len(completion) :]
                completion = data["completion"]
                if run_manager:
                    await run_manager.on_llm_new_token(
                        delta,
                    )
        else:
            response = await self.client.acompletion(**params)
            completion = response["completion"]
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])
import asyncio
import inspect
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence

import langchain
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain.schema import (
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    LLMResult,
    PromptValue,
)
from pydantic import Extra, Field, root_validator


def _get_verbosity() -> bool:
    return langchain.verbose


class BaseChatModel(BaseLanguageModel, ABC):
    verbose: bool = Field(default_factory=_get_verbosity)
    """Whether to print out response text."""
    callbacks: Callbacks = Field(default=None, exclude=True)
    callback_manager: Optional[BaseCallbackManager] = Field(default=None, exclude=True)

    @root_validator()
    def raise_deprecation(cls, values: Dict) -> Dict:
        """Raise deprecation warning if callback_manager is used."""
        if values.get("callback_manager") is not None:
            warnings.warn(
                "callback_manager is deprecated. Please use callbacks instead.",
                DeprecationWarning,
            )
            values["callbacks"] = values.pop("callback_manager", None)
        return values

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        return {}

    def generate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Top Level call"""

        params = self.dict()
        params["stop"] = stop

        callback_manager = CallbackManager.configure(callbacks, self.callbacks, self.verbose)
        run_manager = callback_manager.on_chat_model_start(
            {"name": self.__class__.__name__}, messages, invocation_params=params
        )

        new_arg_supported = inspect.signature(self._generate).parameters.get("run_manager")
        try:
            results = [
                self._generate(m, stop=stop, run_manager=run_manager)
                if new_arg_supported
                else self._generate(m, stop=stop)
                for m in messages
            ]
        except (KeyboardInterrupt, Exception) as e:
            run_manager.on_llm_error(e)
            raise e
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
        run_manager.on_llm_end(output)
        return output

    async def agenerate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Top Level call"""
        params = self.dict()
        params["stop"] = stop

        callback_manager = AsyncCallbackManager.configure(callbacks, self.callbacks, self.verbose)
        run_manager = await callback_manager.on_chat_model_start(
            {"name": self.__class__.__name__}, messages, invocation_params=params
        )

        new_arg_supported = inspect.signature(self._agenerate).parameters.get("run_manager")
        try:
            results = await asyncio.gather(
                *[
                    self._agenerate(m, stop=stop, run_manager=run_manager)
                    if new_arg_supported
                    else self._agenerate(m, stop=stop)
                    for m in messages
                ]
            )
        except (KeyboardInterrupt, Exception) as e:
            await run_manager.on_llm_error(e)
            raise e
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
        await run_manager.on_llm_end(output)
        return output

    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        return self.generate(prompt_messages, stop=stop, callbacks=callbacks)

    async def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        return await self.agenerate(prompt_messages, stop=stop, callbacks=callbacks)

    @abstractmethod
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        """Top Level call"""

    @abstractmethod
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        """Top Level call"""

    def __call__(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> BaseMessage:
        generation = self.generate([messages], stop=stop, callbacks=callbacks).generations[0][0]
        if isinstance(generation, ChatGeneration):
            return generation.message
        else:
            raise ValueError("Unexpected generation type")

    def call_as_llm(self, message: str, stop: Optional[List[str]] = None) -> str:
        return self.predict(message, stop=stop)

    def predict(self, text: str, *, stop: Optional[Sequence[str]] = None) -> str:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        result = self([HumanMessage(content=text)], stop=_stop)
        return result.content

    def predict_messages(self, messages: List[BaseMessage], *, stop: Optional[Sequence[str]] = None) -> BaseMessage:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        return self(messages, stop=_stop)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """Return type of chat model."""

    def dict(self, **kwargs: Any) -> Dict:
        """Return a dictionary of the LLM."""
        starter_dict = dict(self._identifying_params)
        starter_dict["_type"] = self._llm_type
        return starter_dict
"""Azure OpenAI chat wrapper."""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

from pydantic import root_validator

from real_agents.adapters.models.openai import ChatOpenAI
from langchain.schema import ChatResult
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class AzureChatOpenAI(ChatOpenAI):
    """Wrapper around Azure OpenAI Chat Completion API. To use this class you
    must have a deployed model on Azure OpenAI. Use `deployment_name` in the
    constructor to refer to the "Model deployment name" in the Azure portal.

    In addition, you should have the ``openai`` python package installed, and the
    following environment variables set or passed in constructor in lower case:
    - ``OPENAI_API_TYPE`` (default: ``azure``)
    - ``OPENAI_API_KEY``
    - ``OPENAI_API_BASE``
    - ``OPENAI_API_VERSION``

    For exmaple, if you have `gpt-35-turbo` deployed, with the deployment name
    `35-turbo-dev`, the constructor should look like:

    .. code-block:: python
        AzureChatOpenAI(
            deployment_name="35-turbo-dev",
            openai_api_version="2023-03-15-preview",
        )

    Be aware the API version may change.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.
    """

    deployment_name: str = ""
    openai_api_type: str = "azure"
    openai_api_base: str = ""
    openai_api_version: str = ""
    openai_api_key: str = ""
    openai_organization: str = ""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values,
            "openai_api_key",
            "OPENAI_API_KEY",
        )
        openai_api_base = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
        )
        openai_api_version = get_from_dict_or_env(
            values,
            "openai_api_version",
            "OPENAI_API_VERSION",
        )
        openai_api_type = get_from_dict_or_env(
            values,
            "openai_api_type",
            "OPENAI_API_TYPE",
        )
        openai_organization = get_from_dict_or_env(
            values,
            "openai_organization",
            "OPENAI_ORGANIZATION",
            default="",
        )
        try:
            import openai

            openai.api_type = openai_api_type
            openai.api_base = openai_api_base
            openai.api_version = openai_api_version
            openai.api_key = openai_api_key
            if openai_organization:
                openai.organization = openai_organization
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        try:
            values["client"] = openai.ChatCompletion
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            **super()._default_params,
            "engine": self.deployment_name,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**self._default_params}

    @property
    def _llm_type(self) -> str:
        return "azure-openai-chat"

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        for res in response["choices"]:
            if res.get("finish_reason", None) == "content_filter":
                raise ValueError(
                    "Azure has not provided the response due to a content"
                    " filter being triggered"
                )
        return super()._create_chat_result(response)
"""OpenAI chat wrapper."""
from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from pydantic import Extra, Field, root_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def _create_retry_decorator(llm: ChatOpenAI) -> Callable[[Any], Any]:
    import openai

    min_seconds = 1
    max_seconds = 60
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
                retry_if_exception_type(openai.error.Timeout)
                | retry_if_exception_type(openai.error.APIError)
                | retry_if_exception_type(openai.error.APIConnectionError)
                | retry_if_exception_type(openai.error.RateLimitError)
                | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


async def acompletion_with_retry(llm: ChatOpenAI, **kwargs: Any) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        # Use OpenAI's async api https://github.com/openai/openai-python#async-api
        return await llm.client.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)


def _convert_dict_to_message(_dict: dict) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(content=_dict["content"])
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


class ChatOpenAI(BaseChatModel):
    """Wrapper around OpenAI Chat large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatOpenAI
            openai = ChatOpenAI(model_name="gpt-3.5-turbo")
    """

    client: Any  #: :meta private:
    model_name: str = "gpt-3.5-turbo"
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    openai_api_key: Optional[str] =  'sk-q5qfXMgYm3K8F1nxGnJ1yJjMUwzoQJlVZp2bZrIGY7nEdy2E'
    """Base URL path for API requests,
    leave blank if not using a proxy or service emulator."""
    openai_api_base: Optional[str] = 'https://api.chatanywhere.tech/v1'
    openai_organization: Optional[str] = None
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to OpenAI completion API. Default is 600 seconds."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    streaming: bool = False
    """Whether to stream the results or not."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    stop: Optional[List[str]] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        disallowed_model_kwargs = all_required_field_names | {"model"}
        invalid_model_kwargs = disallowed_model_kwargs.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_organization = get_from_dict_or_env(
            values,
            "openai_organization",
            "OPENAI_ORGANIZATION",
            default="",
        )
        openai_api_base = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
            default="",
        )

        try:
            import openai

        except ImportError:
            raise ValueError(
                "Could not import openai python package. " "Please install it with `pip install openai`.")

        if openai_organization:
            openai.organization = openai_organization
        if openai_api_base:
            openai.api_base = openai_api_base

        try:
            values["client"] = openai.ChatCompletion
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model_name,
            "request_timeout": self.request_timeout,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

    def _create_retry_decorator(self) -> Callable[[Any], Any]:
        import openai

        min_seconds = 1
        max_seconds = 60
        # Wait 2^x * 1 second between each retry starting with
        # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
        return retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
            retry=(
                    retry_if_exception_type(openai.error.Timeout)
                    | retry_if_exception_type(openai.error.APIError)
                    | retry_if_exception_type(openai.error.APIConnectionError)
                    | retry_if_exception_type(openai.error.RateLimitError)
                    | retry_if_exception_type(openai.error.ServiceUnavailableError)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

    def completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.client.create(**kwargs)

        return _completion_with_retry(**kwargs)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        return {"token_usage": overall_token_usage, "model_name": self.model_name}

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        import openai

        if self.openai_api_key:
            import os

            # Use the pass-in key, if the user provides
            openai.api_key = self.openai_api_key
        else:
            # Use the environment variable if neither is provided
            import os

            openai_api_key = os.environ.get("OPENAI_API_KEY", None)
            openai.api_key = openai_api_key

        if self.stop is not None:
            if stop is None:
                stop = self.stop
            else:
                stop.extend(self.stop)

        message_dicts, params = self._create_message_dicts(messages, stop)
        if self.streaming:
            inner_completion = ""
            default_role = "assistant"
            params["stream"] = True
            for stream_resp in self.completion_with_retry(messages=message_dicts,
                                                          **params):
                role = stream_resp["choices"][0]["delta"].get("role", default_role)
                if role is None:
                    role = default_role
                token = stream_resp["choices"][0]["delta"].get("content", "")
                inner_completion += token
                if run_manager:
                    run_manager.on_llm_new_token(token)
            message = _convert_dict_to_message(
                {"content": inner_completion, "role": role})
            return ChatResult(generations=[ChatGeneration(message=message)])
        response = self.completion_with_retry(messages=message_dicts, **params)
        return self._create_chat_result(response)

    def _create_message_dicts(
            self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params: Dict[str, Any] = {**{"model": self.model_name}, **self._default_params}
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            gen = ChatGeneration(message=message)
            generations.append(gen)
        llm_output = {"token_usage": response["usage"], "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        import os

        import openai

        openai_api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        openai_api_key = os.environ.get("OPENAI_API_KEY", None)
        openai.api_key = openai_api_key
        openai.api_base = openai_api_base

        message_dicts, params = self._create_message_dicts(messages, stop)
        if self.streaming:
            inner_completion = ""
            role = "assistant"
            params["stream"] = True
            async for stream_resp in await acompletion_with_retry(self,
                                                                  messages=message_dicts,
                                                                  **params):
                role = stream_resp["choices"][0]["delta"].get("role", role)
                token = stream_resp["choices"][0]["delta"].get("content", "")
                inner_completion += token
                if run_manager:
                    await run_manager.on_llm_new_token(token)
            message = _convert_dict_to_message(
                {"content": inner_completion, "role": role})
            return ChatResult(generations=[ChatGeneration(message=message)])
        else:
            response = await acompletion_with_retry(self, messages=message_dicts,
                                                    **params)
            return self._create_chat_result(response)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "openai-chat"

    def get_num_tokens(self, text: str) -> int:
        """Calculate num tokens with tiktoken package."""
        # tiktoken NOT supported for Python 3.7 or below
        if sys.version_info[1] <= 7:
            return super().get_num_tokens(text)
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please install it with `pip install tiktoken`."
            )
        # create a GPT-3.5-Turbo encoder instance
        enc = tiktoken.encoding_for_model(self.model_name)

        # encode the text using the GPT-3.5-Turbo encoder
        tokenized_text = enc.encode(text)

        # calculate the number of tokens in the encoded text
        return len(tokenized_text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Calculate num tokens for gpt-3.5-turbo and gpt-4 with tiktoken package.

        Official documentation: https://github.com/openai/openai-cookbook/blob/
        main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb"""
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please install it with `pip install tiktoken`."
            )

        model = self.model_name
        if model == "gpt-3.5-turbo":
            # gpt-3.5-turbo may change over time.
            # Returning num tokens assuming gpt-3.5-turbo-0301.
            model = "gpt-3.5-turbo-0301"
        elif model == "gpt-4":
            # gpt-4 may change over time.
            # Returning num tokens assuming gpt-4-0314.
            model = "gpt-4-0314"

        # Returns the number of tokens used by a list of messages.
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        if model == "gpt-3.5-turbo-0301":
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_message = 4
            # if there's a name, the role is omitted
            tokens_per_name = -1
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"get_num_tokens_from_messages() is not presently implemented "
                f"for model {model}."
                "See https://github.com/openai/openai-python/blob/main/chatml.md for "
                "information on how messages are converted to tokens."
            )
        num_tokens = 0
        messages_dict = [_convert_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        # every reply is primed with <im_start>assistant
        num_tokens += 3
        return num_tokens
from langchain.chat_models.google_palm import ChatGooglePalm

from real_agents.adapters.models.anthropic import ChatAnthropic
from real_agents.adapters.models.openai import ChatOpenAI
from real_agents.adapters.models.azure_openai import AzureChatOpenAI

__all__ = [
    "ChatOpenAI",
    "ChatAnthropic",
    "ChatGooglePalm",
    "AzureChatOpenAI",
]

type_to_cls_dict = {
    "chat_anthropic": ChatAnthropic,
    "chat_google_palm": ChatGooglePalm,
    "chat_openai": ChatOpenAI,
    "azure_chat_openai": AzureChatOpenAI,
}
