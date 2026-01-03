from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from langchain_core.documents import Document
import logging

from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from .relevance_checker import RelevanceChecker

logger = logging.getLogger(__name__)


# ---------------------------
# Agent State Definition
# ---------------------------
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: Any  # 🔥 Generic to support custom hybrid retrievers


# ---------------------------
# Workflow Class
# ---------------------------
class AgentWorkflow:
    def __init__(self):
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.relevance_checker = RelevanceChecker()
        self.compiled_workflow = self.build_workflow()

    # ---------------------------
    # Build LangGraph Workflow
    # ---------------------------
    def build_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)

        workflow.set_entry_point("check_relevance")

        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {
                "relevant": "research",
                "irrelevant": END
            }
        )

        workflow.add_edge("research", "verify")

        workflow.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {
                "re_research": "research",
                "end": END
            }
        )

        return workflow.compile()

    # ---------------------------
    # Relevance Check
    # ---------------------------
    def _check_relevance_step(self, state: AgentState) -> Dict:
        retriever = state["retriever"]

        classification = self.relevance_checker.check(
            question=state["question"],
            retriever=retriever,
            k=20
        )

        if classification in ("CAN_ANSWER", "PARTIAL"):
            return {"is_relevant": True}

        return {
            "is_relevant": False,
            "draft_answer": (
                "This question is not related to the uploaded document(s), "
                "or there is insufficient information to answer it."
            )
        }

    def _decide_after_relevance_check(self, state: AgentState) -> str:
        decision = "relevant" if state["is_relevant"] else "irrelevant"
        logger.debug(f"Relevance decision: {decision}")
        return decision

    # ---------------------------
    # Full Pipeline Entry
    # ---------------------------
    def full_pipeline(self, question: str, retriever: Any):
        try:
            logger.info(f"Running pipeline for question: {question}")

            # 🔥 Always use invoke() (LangChain 1.x standard)
            documents = retriever.invoke(question)

            logger.info(f"Retrieved {len(documents)} documents")

            initial_state: AgentState = {
                "question": question,
                "documents": documents,
                "draft_answer": "",
                "verification_report": "",
                "is_relevant": False,
                "retriever": retriever
            }

            final_state = self.compiled_workflow.invoke(initial_state)

            return {
                "draft_answer": final_state.get("draft_answer", ""),
                "verification_report": final_state.get("verification_report", "")
            }

        except Exception as e:
            logger.exception("Workflow execution failed")
            raise e

    # ---------------------------
    # Research Step
    # ---------------------------
    def _research_step(self, state: AgentState) -> Dict:
        logger.debug("Entering research step")
        result = self.researcher.generate(
            question=state["question"],
            documents=state["documents"]
        )
        return {"draft_answer": result["draft_answer"]}

    # ---------------------------
    # Verification Step
    # ---------------------------
    def _verification_step(self, state: AgentState) -> Dict:
        logger.debug("Entering verification step")
        result = self.verifier.check(
            answer=state["draft_answer"],
            documents=state["documents"]
        )
        return {"verification_report": result["verification_report"]}

    # ---------------------------
    # Decide Loop or End
    # ---------------------------
    def _decide_next_step(self, state: AgentState) -> str:
        report = state["verification_report"]

        if "Supported: NO" in report or "Relevant: NO" in report:
            logger.info("Verification failed → re-research")
            return "re_research"

        logger.info("Verification successful → end workflow")
        return "end"
