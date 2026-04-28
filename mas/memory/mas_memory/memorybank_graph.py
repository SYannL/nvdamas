from dataclasses import dataclass
from typing import Any

from mas.langchain_compat import Document

from .memorybank import MemoryBankMASMemory
from ..common import MASMessage
from .entity_graph_module import EntityGraphModule


@dataclass
class MemoryBankGraphMASMemory(MemoryBankMASMemory):
    def __post_init__(self):
        super().__post_init__()

        self.entity_graph = EntityGraphModule(
            persist_dir=self.persist_dir,
            embedding_func=self.embedding_func,
            llm_model=self.llm_model,
            merge_threshold=self.global_config.get("entity_merge_threshold", 0.9),
            persist_every=self.global_config.get("entity_graph_persist_every", 10),
        )
        self.last_saved_message: MASMessage | None = None
        self._global_retriever: "MemoryBankGraphMASMemory" | None = None

    def _add_memory_raw(self, mas_message: MASMessage) -> None:
        raw = mas_message.get_extra_field("raw_task_main") or mas_message.task_main
        meta_data: dict = MASMessage.to_dict(mas_message)
        memory_doc = Document(
            page_content=raw,
            metadata=meta_data
        )
        if mas_message.label == True or mas_message.label == False:
            self.main_memory.add_documents([memory_doc])
        else:
            raise ValueError("The mas_message must have label!")
        self._index_done()

    def set_global_retriever(self, global_retriever: "MemoryBankGraphMASMemory") -> None:
        self._global_retriever = global_retriever

    def add_memory(self, mas_message: MASMessage) -> None:
        if self.global_config.get("freeze_memory", False):
            return
        knowledge_summary = self.entity_graph.summarize_task_knowledge(mas_message)
        if knowledge_summary:
            mas_message.add_extra_field("knowledge_summary", knowledge_summary)

        super().add_memory(mas_message)
        self.last_saved_message = mas_message

        if mas_message.label is True and self._global_retriever is None:
            self.entity_graph.add_task_from_message(mas_message)

    def add_memory_from_peer(self, mas_message: MASMessage, source_id: str | None = None) -> None:
        if source_id:
            mas_message.add_extra_field("source_id", source_id)
            mas_message.add_extra_field("source_scope", "global")

        self._add_memory_raw(mas_message)
        self.last_saved_message = mas_message

        if mas_message.label is True:
            self.entity_graph.add_task_from_message(mas_message)

    def persist_entity_graph(self) -> None:
        """Force persist the entity graph to disk (call after merge loop to ensure global graph is saved)."""
        self.entity_graph.persist()

    def retrieve_memory(
        self,
        query_task: str,
        successful_topk: int = 1,
        failed_topk: int = 1,
        **args
    ) -> tuple[list, list, list]:
        if self._global_retriever is not None:
            return self._global_retriever.retrieve_memory(
                query_task=query_task,
                successful_topk=successful_topk,
                failed_topk=failed_topk,
                **args
            )

        true_msgs, false_msgs, _ = super().retrieve_memory(
            query_task=query_task,
            successful_topk=successful_topk,
            failed_topk=failed_topk,
            **args
        )

        insights = self.entity_graph.retrieve_insights(query_task)
        return true_msgs, false_msgs, insights
