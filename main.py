from typing import Tuple

import asyncio
import semantic_kernel as sk

from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)

from semantic_kernel.connectors.memory.azure_cognitive_search import ( 
    AzureCognitiveSearchMemoryStore,
)

kernel = sk.Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
azure_ai_search_api_key, azure_ai_search_url = sk.azure_aisearch_settings_from_dot_env()

    # next line assumes chat deployment name is "turbo", adjust the deployment name to the value of your chat model if needed
azure_chat_service = AzureChatCompletion(deployment_name="completions-v0", endpoint=endpoint, api_key=api_key)    # next line assumes embeddings deployment name is "text-embedding", adjust the deployment name to the value of your chat model if needed
azure_text_embedding = AzureTextEmbedding(deployment_name="embeddings-v0", endpoint=endpoint, api_key=api_key)

kernel.add_chat_service("chat_completion", azure_chat_service)
kernel.add_text_embedding_generation_service("ada", azure_text_embedding)


# text-embedding-ada-002 uses a 1536-dimensional embedding vector
kernel.register_memory_store(
    memory_store=AzureCognitiveSearchMemoryStore(
        vector_size=1536,
        search_endpoint=azure_ai_search_url,
        admin_key=azure_ai_search_api_key,
    )
)


async def populate_memory():
    certification_files = {}
    certification_files[
        "https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/az-104"
    ] = "Study guide for Exam AZ-104: Microsoft Azure Administrator"
    certification_files[
        "https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/az-204"
    ] = "Study guide for Exam AZ-204: Developing Solutions for Microsoft Azure"
    certification_files[
        "https://learn.microsoft.com/en-us/credentials/certifications/resources/study-guides/az-900"
    ] = "Study guide for Exam AZ-900: Microsoft Azure Fundamentals"

    memory_collection_name = "SKCertifications"
    print("Adding some Azure Study guide URLs and their descriptions to a volatile Semantic Memory.")

    for i, (entry, value) in enumerate(certification_files.items()):
        await kernel.memory.save_reference(
            collection=memory_collection_name,
            description=value,
            text=value,
            external_id=entry,
            external_source_name="Microsoft Learn",
        )

        print("  URL {} saved".format(i + 1))

async def main():
    await populate_memory()
    ask = "I'm stufying for the AZ-104 certification."
    print("===========================\n" + "Query: " + ask + "\n")

    memory_collection_name = "SKCertifications"
    memories = await kernel.memory.search(memory_collection_name, ask, limit=5, min_relevance_score=0.77)

    for i, memory in enumerate(memories):
        print(f"Result {i + 1}:")
        print("  URL:     : " + memory.id)
        print("  Title    : " + memory.description)
        print("  Relevance: " + str(memory.relevance))
        print()

asyncio.run(main())

