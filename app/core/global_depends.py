from dependency_injector import containers, providers

from app.core.config import settings
from app.repositories.document_embeddings_repository import (
    DocumentEmbeddingsRepository,
)


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    document_embeddings_repository = providers.Singleton(
        DocumentEmbeddingsRepository,
        pg_connection_string=settings.pg_connection_string,
        collection_name=settings.collection_name,
        vertex_project_id=settings.project_id,
        vertex_location=settings.location,
        service_account_file=settings.service_account_json,
    )
