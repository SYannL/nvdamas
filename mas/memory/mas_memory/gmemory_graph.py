from dataclasses import dataclass

from .GMemory import GMemory
from ..common import MASMessage
from .entity_graph_module import EntityGraphModule


@dataclass
class GMemoryGraphMASMemory(GMemory):
    def __post_init__(self):
        super().__post_init__()

        self.entity_graph = EntityGraphModule(
            persist_dir=self.persist_dir,
            embedding_func=self.embedding_func,
            llm_model=self.llm_model,
            merge_threshold=self.global_config.get("entity_merge_threshold", 0.9),
            persist_every=self.global_config.get("entity_graph_persist_every", 10),
        )

    def add_memory(self, mas_message: MASMessage) -> None:
        knowledge_summary = self.entity_graph.summarize_task_knowledge(mas_message)
        if knowledge_summary:
            mas_message.add_extra_field("knowledge_summary", knowledge_summary)
        super().add_memory(mas_message)
        if mas_message.label is True:
            self.entity_graph.add_task_from_message(mas_message)

    def add_memory_with_source(self, mas_message: MASMessage, source_id: str, raw: bool = True) -> None:
        knowledge_summary = self.entity_graph.summarize_task_knowledge(mas_message)
        if knowledge_summary:
            mas_message.add_extra_field("knowledge_summary", knowledge_summary)
        super().add_memory_with_source(mas_message, source_id=source_id, raw=raw)
        if mas_message.label is True:
            self.entity_graph.add_task_from_message(mas_message)

    def retrieve_memory(self, *args, **kwargs):
        true_msgs, false_msgs, insights = super().retrieve_memory(*args, **kwargs)
        query_task = kwargs.get("query_task")
        if query_task is None and len(args) > 0:
            query_task = args[0]
        if query_task:
            graph_insights = self.entity_graph.retrieve_insights(query_task)
            insights = insights + graph_insights
        return true_msgs, false_msgs, insights
