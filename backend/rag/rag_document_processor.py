import json
import logging
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGDocumentProcessor:
    def __init__(self, source_name: str = "GraphQL API"):
        """
        Initialize the RAG document processor
        
        Args:
            source_name: Name of the data source for metadata
        """
        self.source_name = source_name
        self.processed_documents = []
        self.metadata = {
            "source": source_name,
            "processed_at": datetime.now().isoformat(),
            "document_count": 0,
            "entity_types": set()
        }
    
    def process_graphql_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process GraphQL extracted data into RAG-suitable documents
        
        Args:
            data: The extracted GraphQL data
            
        Returns:
            List of processed documents ready for RAG ingestion
        """
        documents = []
        
        for field_name, field_data in data.items():
            if self._is_error_or_skipped(field_data):
                logger.warning(f"Skipping {field_name}: {field_data}")
                continue
                
            logger.info(f"Processing field: {field_name}")
            
            if isinstance(field_data, dict):
                if "results" in field_data and isinstance(field_data["results"], list):
                    # This is a paginated collection (like characters, locations, episodes)
                    collection_doc = self._process_collection(field_data, field_name)
                    if collection_doc:
                        documents.append(collection_doc)
                    
                    # Process individual items in the collection
                    for i, item in enumerate(field_data["results"]):
                        if isinstance(item, dict):
                            item_id = self._get_entity_id(item, f"{field_name}_{i}")
                            doc = self._process_individual_entity(item, field_name, item_id)
                            if doc:
                                documents.append(doc)
                else:
                    # Single entity
                    doc = self._process_individual_entity(field_data, field_name, field_name)
                    if doc:
                        documents.append(doc)
            elif isinstance(field_data, list):
                # Handle list of entities
                for i, item in enumerate(field_data):
                    if isinstance(item, dict):
                        item_id = self._get_entity_id(item, f"{field_name}_{i}")
                        doc = self._process_individual_entity(item, field_name, item_id)
                        if doc:
                            documents.append(doc)
            else:
                # Handle primitive values
                doc = self._create_simple_document(field_name, field_data)
                documents.append(doc)
        
        self.processed_documents.extend(documents)
        self.metadata["document_count"] = len(self.processed_documents)
        
        return documents
    
    def _is_error_or_skipped(self, data: Any) -> bool:
        """Check if data represents an error or skipped field"""
        if isinstance(data, dict):
            return "error" in data or "skipped" in data
        return False
    
    def _get_entity_id(self, entity: Dict[str, Any], fallback_id: str) -> str:
        """Extract entity ID with fallback"""
        if "id" in entity:
            typename = entity.get("__typename", "entity")
            return f"{typename.lower()}_{entity['id']}"
        elif "name" in entity:
            typename = entity.get("__typename", "entity")
            # Clean name for ID
            clean_name = "".join(c for c in entity["name"] if c.isalnum() or c in " -_").replace(" ", "_").lower()
            return f"{typename.lower()}_{clean_name}"
        else:
            return fallback_id
    
    def _process_collection(self, collection_data: Dict[str, Any], collection_name: str) -> Optional[Dict[str, Any]]:
        """Process a collection overview document"""
        typename = collection_data.get("__typename", collection_name.title())
        self.metadata["entity_types"].add(typename)
        
        content_parts = [f"# {typename} Collection Overview"]
        
        # Add info section
        if "info" in collection_data:
            info = collection_data["info"]
            content_parts.append("\n## Collection Statistics")
            for key, value in info.items():
                if not key.startswith("__") and value is not None:
                    content_parts.append(f"- **{self._format_field_name(key)}**: {value}")
        
        # Add summary of results
        if "results" in collection_data and collection_data["results"]:
            results = collection_data["results"]
            content_parts.append(f"\n## Available {typename}")
            content_parts.append(f"This collection contains {len(results)} {typename.lower()} entries:")
            
            for item in results[:20]:  # Show first 20
                if isinstance(item, dict):
                    name = item.get("name") or item.get("title") or item.get("id", "Unknown")
                    item_type = item.get("__typename", typename.rstrip('s'))
                    content_parts.append(f"- {name} ({item_type})")
            
            if len(results) > 20:
                content_parts.append(f"... and {len(results) - 20} more entries")
        
        return {
            "id": f"{collection_name}_collection",
            "content": "\n".join(content_parts),
            "metadata": {
                "id": f"{collection_name}_collection",
                "type": f"{typename}_Collection",
                "source": self.source_name,
                "entity_type": collection_name,
                "collection_size": len(collection_data.get("results", [])),
                "keywords": [collection_name.lower(), "collection", "overview", typename.lower()]
            },
            "title": f"{typename} Collection",
            "type": "collection_document"
        }
    
    def _process_individual_entity(self, entity: Dict[str, Any], entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Process a single entity into a detailed RAG document
        
        Args:
            entity: The entity data
            entity_type: Type of entity (e.g., 'characters', 'locations')
            entity_id: Unique identifier for the entity
            
        Returns:
            Processed document or None if processing fails
        """
        if not isinstance(entity, dict):
            return None
            
        # Extract typename if available
        typename = entity.get("__typename", entity_type.rstrip('s'))
        self.metadata["entity_types"].add(typename)
        
        # Get entity name/title
        entity_name = entity.get("name") or entity.get("title") or entity.get("id", "Unknown")
        
        # Create the main document content
        content_parts = []
        metadata = {
            "id": entity_id,
            "type": typename,
            "source": self.source_name,
            "entity_type": entity_type,
            "entity_name": entity_name
        }
        
        # Build the main content with title
        title = f"{entity_name} ({typename})"
        content_parts.append(f"# {title}")
        content_parts.append(f"**Type**: {typename}")
        content_parts.append(f"**Collection**: {entity_type}")
        
        # Process all fields systematically
        simple_fields = {}
        complex_fields = {}
        relationships = {}
        
        for key, value in entity.items():
            if key.startswith("__"):
                continue  # Skip GraphQL metadata
                
            if value is None or value == "":
                continue  # Skip empty values
                
            if isinstance(value, (str, int, float, bool)):
                simple_fields[key] = value
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    # List of objects (relationships)
                    relationships[key] = value
                else:
                    # List of primitives
                    simple_fields[key] = value
            elif isinstance(value, dict):
                # Single object relationship
                relationships[key] = value
            else:
                simple_fields[key] = value
        
        # Add basic information section
        if simple_fields:
            content_parts.append("\n## Basic Information")
            for key, value in simple_fields.items():
                formatted_key = self._format_field_name(key)
                if isinstance(value, list):
                    if len(value) <= 5:
                        content_parts.append(f"- **{formatted_key}**: {', '.join(map(str, value))}")
                    else:
                        content_parts.append(f"- **{formatted_key}**: {', '.join(map(str, value[:5]))} (and {len(value)-5} more)")
                else:
                    content_parts.append(f"- **{formatted_key}**: {value}")
        
        # Add detailed relationship sections
        if relationships:
            for rel_name, rel_data in relationships.items():
                section_title = self._format_field_name(rel_name)
                content_parts.append(f"\n## {section_title}")
                
                if isinstance(rel_data, list):
                    if len(rel_data) > 0:
                        content_parts.append(f"Associated {rel_name} ({len(rel_data)} total):")
                        for i, item in enumerate(rel_data):
                            if i >= 15:  # Limit to avoid huge documents
                                content_parts.append(f"... and {len(rel_data) - 15} more {rel_name}")
                                break
                            item_summary = self._create_detailed_item_summary(item, rel_name)
                            if item_summary:
                                content_parts.append(f"- {item_summary}")
                    else:
                        content_parts.append(f"No {rel_name} available.")
                elif isinstance(rel_data, dict):
                    item_summary = self._create_detailed_item_summary(rel_data, rel_name)
                    if item_summary:
                        content_parts.append(item_summary)
        
        # Create comprehensive metadata
        metadata.update({
            "simple_fields": simple_fields,
            "relationship_fields": list(relationships.keys()),
            "relationship_count": len(relationships),
            "total_fields": len(simple_fields) + len(relationships),
            "has_relationships": len(relationships) > 0
        })
        
        # Add comprehensive searchable keywords
        keywords = self._extract_comprehensive_keywords(entity, typename, entity_name)
        metadata["keywords"] = keywords
        
        document = {
            "id": entity_id,
            "content": "\n".join(content_parts),
            "metadata": metadata,
            "title": title,
            "type": "entity_document"
        }
        
        return document
    
    def _format_field_name(self, field_name: str) -> str:
        """Format field name for display"""
        # Convert camelCase to Title Case
        import re
        formatted = re.sub(r'([A-Z])', r' \1', field_name)
        return formatted.strip().title()
    
    def _create_detailed_item_summary(self, item: Any, relation_type: str) -> str:
        """Create a detailed summary string for a relationship item"""
        if isinstance(item, dict):
            # Extract key information
            name = item.get("name") or item.get("title") or item.get("id")
            type_info = item.get("__typename", "")
            
            summary_parts = []
            
            # Add name/title
            if name:
                if type_info:
                    summary_parts.append(f"**{name}** ({type_info})")
                else:
                    summary_parts.append(f"**{name}**")
            
            # Add additional context based on type
            additional_info = []
            
            # For episodes
            if "episode" in item:
                additional_info.append(f"Episode: {item['episode']}")
            if "air_date" in item:
                additional_info.append(f"Aired: {item['air_date']}")
            
            # For locations
            if "dimension" in item:
                additional_info.append(f"Dimension: {item['dimension']}")
            if "type" in item and item["type"]:
                additional_info.append(f"Type: {item['type']}")
            
            # For characters
            if "status" in item:
                additional_info.append(f"Status: {item['status']}")
            if "species" in item:
                additional_info.append(f"Species: {item['species']}")
            if "gender" in item:
                additional_info.append(f"Gender: {item['gender']}")
            
            # Add other relevant fields
            for key in ["code", "capital", "currency", "native", "description"]:
                if key in item and item[key] and item[key] != name:
                    additional_info.append(f"{key.title()}: {item[key]}")
            
            if additional_info:
                summary_parts.append(f" - {', '.join(additional_info[:5])}")  # Limit to 5 additional fields
            
            return "".join(summary_parts) if summary_parts else str(item)
        else:
            return str(item)
    
    def _extract_comprehensive_keywords(self, entity: Dict[str, Any], typename: str, entity_name: str) -> List[str]:
        """Extract comprehensive searchable keywords from entity"""
        keywords = [typename.lower(), entity_name.lower()]
        
        # Add words from entity name
        if entity_name:
            keywords.extend(entity_name.lower().split())
        
        # Add values from key fields
        key_fields = ["name", "code", "title", "native", "capital", "currency", "status", "species", "gender", "type", "dimension", "episode"]
        for field in key_fields:
            if field in entity and entity[field]:
                value = str(entity[field])
                keywords.extend(value.lower().split())
        
        # Add relationship types
        for key, value in entity.items():
            if isinstance(value, (list, dict)) and not key.startswith("__"):
                keywords.append(key.lower())
                
                # Add names from related objects
                if isinstance(value, list):
                    for item in value[:3]:  # Limit to first 3 items
                        if isinstance(item, dict) and "name" in item:
                            keywords.extend(item["name"].lower().split())
                elif isinstance(value, dict) and "name" in value:
                    keywords.extend(value["name"].lower().split())
        
        # Clean and deduplicate keywords
        cleaned_keywords = []
        for keyword in keywords:
            # Remove special characters and split compound words
            clean_word = "".join(c for c in keyword if c.isalnum() or c.isspace()).strip()
            if clean_word and len(clean_word) > 1:
                cleaned_keywords.extend(clean_word.split())
        
        return list(set(cleaned_keywords))  # Remove duplicates
    
    def _create_simple_document(self, field_name: str, data: Any) -> Dict[str, Any]:
        """Create a document for simple field data"""
        content = f"# {self._format_field_name(field_name)}\n\n{data}"
        
        return {
            "id": field_name,
            "content": content,
            "metadata": {
                "id": field_name,
                "type": "simple_field",
                "source": self.source_name,
                "field_name": field_name,
                "data_type": type(data).__name__,
                "keywords": [field_name.lower(), "simple", "field"]
            },
            "title": self._format_field_name(field_name),
            "type": "simple_document"
        }
    
    def create_relationship_documents(self) -> List[Dict[str, Any]]:
        """Create additional documents focusing on relationships between entities"""
        relationship_docs = []
        
        # Group entities by type
        entities_by_type = {}
        for doc in self.processed_documents:
            if doc["type"] == "entity_document":
                entity_type = doc["metadata"]["type"]
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(doc)
        
        # Create cross-reference documents
        relationship_docs.extend(self._create_cross_reference_documents())
        
        # Create entity type summaries
        for entity_type, docs in entities_by_type.items():
            if len(docs) > 1:
                rel_doc = self._create_entity_type_summary(entity_type, docs)
                relationship_docs.append(rel_doc)
        
        return relationship_docs
    
    def _create_cross_reference_documents(self) -> List[Dict[str, Any]]:
        """Create documents that show relationships between different entity types"""
        cross_ref_docs = []
        
        # Find character-episode relationships
        characters = [doc for doc in self.processed_documents if doc["metadata"].get("type") == "Character"]
        episodes = [doc for doc in self.processed_documents if doc["metadata"].get("type") == "Episode"]
        
        if characters and episodes:
            # Create character-episode cross-reference
            content_parts = ["# Character-Episode Relationships"]
            content_parts.append("This document shows which characters appear in which episodes.")
            
            for char_doc in characters[:20]:  # Limit to avoid huge documents
                char_name = char_doc["metadata"].get("entity_name", "Unknown")
                # Extract episode information from character content
                char_episodes = self._extract_episodes_from_content(char_doc["content"])
                if char_episodes:
                    content_parts.append(f"\n## {char_name}")
                    content_parts.append(f"Appears in {len(char_episodes)} episodes:")
                    for ep in char_episodes[:10]:
                        content_parts.append(f"- {ep}")
            
            cross_ref_docs.append({
                "id": "character_episode_relationships",
                "content": "\n".join(content_parts),
                "metadata": {
                    "id": "character_episode_relationships",
                    "type": "cross_reference",
                    "source": self.source_name,
                    "relationship_types": ["Character", "Episode"],
                    "keywords": ["character", "episode", "relationships", "appearances"]
                },
                "title": "Character-Episode Relationships",
                "type": "relationship_document"
            })
        
        return cross_ref_docs
    
    def _extract_episodes_from_content(self, content: str) -> List[str]:
        """Extract episode names from character content"""
        episodes = []
        lines = content.split('\n')
        in_episode_section = False
        
        for line in lines:
            if "## Episode" in line:
                in_episode_section = True
                continue
            elif line.startswith("## ") and in_episode_section:
                break
            elif in_episode_section and line.startswith("- **"):
                # Extract episode name
                import re
                match = re.search(r'\*\*(.*?)\*\*', line)
                if match:
                    episodes.append(match.group(1))
        
        return episodes
    
    def _create_entity_type_summary(self, entity_type: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a comprehensive summary document for all entities of a specific type"""
        content_parts = [f"# {entity_type} Summary"]
        content_parts.append(f"Comprehensive overview of all {entity_type} entities in the database.")
        content_parts.append(f"\n**Total {entity_type} entities**: {len(docs)}")
        
        # Collect field statistics
        all_simple_fields = set()
        all_relationship_fields = set()
        field_usage = {}
        
        for doc in docs:
            simple_fields = doc["metadata"].get("simple_fields", {})
            relationship_fields = doc["metadata"].get("relationship_fields", [])
            
            all_simple_fields.update(simple_fields.keys())
            all_relationship_fields.update(relationship_fields)
            
            # Count field usage
            for field in list(simple_fields.keys()) + relationship_fields:
                field_usage[field] = field_usage.get(field, 0) + 1
        
        # Add field information
        if all_simple_fields:
            content_parts.append(f"\n## Available Simple Fields")
            sorted_simple = sorted(all_simple_fields)
            for field in sorted_simple[:15]:  # Limit display
                usage_count = field_usage.get(field, 0)
                usage_percent = (usage_count / len(docs)) * 100
                content_parts.append(f"- **{self._format_field_name(field)}**: Used in {usage_count}/{len(docs)} entities ({usage_percent:.1f}%)")
        
        if all_relationship_fields:
            content_parts.append(f"\n## Available Relationship Fields")
            sorted_rel = sorted(all_relationship_fields)
            for field in sorted_rel[:10]:  # Limit display
                usage_count = field_usage.get(field, 0)
                usage_percent = (usage_count / len(docs)) * 100
                content_parts.append(f"- **{self._format_field_name(field)}**: Used in {usage_count}/{len(docs)} entities ({usage_percent:.1f}%)")
        
        # Add sample entities with more detail
        content_parts.append(f"\n## Sample {entity_type} Entities")
        for i, doc in enumerate(docs[:10]):  # Show first 10
            entity_name = doc["metadata"].get("entity_name", "Unknown")
            simple_field_count = len(doc["metadata"].get("simple_fields", {}))
            rel_count = doc["metadata"].get("relationship_count", 0)
            content_parts.append(f"- **{entity_name}** - {simple_field_count} fields, {rel_count} relationships")
        
        if len(docs) > 10:
            content_parts.append(f"... and {len(docs) - 10} more {entity_type.lower()} entities")
        
        return {
            "id": f"{entity_type.lower()}_summary",
            "content": "\n".join(content_parts),
            "metadata": {
                "id": f"{entity_type.lower()}_summary",
                "type": "entity_type_summary",
                "source": self.source_name,
                "entity_type": entity_type,
                "entity_count": len(docs),
                "simple_field_count": len(all_simple_fields),
                "relationship_field_count": len(all_relationship_fields),
                "keywords": [entity_type.lower(), "summary", "overview", "statistics", "fields"]
            },
            "title": f"{entity_type} Summary",
            "type": "summary_document"
        }
    
    def save_rag_documents(self, output_path: str = "rag_documents.json"):
        """Save processed documents in RAG-ready format"""
        all_docs = self.processed_documents + self.create_relationship_documents()
        
        output = {
            "metadata": {**self.metadata, "entity_types": list(self.metadata["entity_types"])},
            "documents": all_docs,
            "total_documents": len(all_docs)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(all_docs)} RAG documents to {output_path}")
        return output_path
    
    def save_text_documents(self, output_dir: str = "rag_texts"):
        """Save each document as a separate text file for easy ingestion"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        all_docs = self.processed_documents + self.create_relationship_documents()
        
        for doc in all_docs:
            # Create safe filename
            safe_id = "".join(c for c in doc['id'] if c.isalnum() or c in "_-").lower()
            filename = f"{safe_id}.txt"
            filepath = output_path / filename
            
            # Create content with metadata header
            content_lines = [
                f"Title: {doc['title']}",
                f"Type: {doc['type']}",
                f"Source: {doc['metadata']['source']}",
                f"ID: {doc['id']}",
                f"Keywords: {', '.join(doc['metadata'].get('keywords', []))}",
                "=" * 50,
                "",
                doc['content']
            ]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content_lines))
        
        logger.info(f"Saved {len(all_docs)} text documents to {output_dir}/")
        return output_dir


def main():
    parser = argparse.ArgumentParser(description='Process GraphQL extracted data for RAG ingestion')
    parser.add_argument('input_file', help='Path to the extracted GraphQL data JSON file')
    parser.add_argument('--output', '-o', default='/home/genesys/voicebot/Apollo-Voicebot/backend/rag/knowledge_base/rag_documents.json', help='Output JSON file for RAG documents')
    parser.add_argument('--text-dir', default='backend/rag/knowledge_base/', help='Directory for individual text files')
    parser.add_argument('--source-name', default='GraphQL API', help='Name of the data source')
    
    args = parser.parse_args()
    
    try:
        # Load the extracted data
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded data from {args.input_file}")
        
        # Process the data
        processor = RAGDocumentProcessor(args.source_name)
        documents = processor.process_graphql_data(data)
        
        logger.info(f"Processed {len(documents)} documents")
        
        # Save in both formats
        json_output = processor.save_rag_documents(args.output)
        text_output = processor.save_text_documents(args.text_dir)
        
        # Print detailed summary
        print(f"\n=== Processing Complete ===")
        print(f"Input file: {args.input_file}")
        print(f"JSON output: {json_output}")
        print(f"Text files: {text_output}")
        print(f"Total documents: {len(processor.processed_documents) + len(processor.create_relationship_documents())}")
        print(f"Entity types found: {', '.join(processor.metadata['entity_types'])}")
        
        # Count documents by type
        doc_types = {}
        all_docs = processor.processed_documents + processor.create_relationship_documents()
        for doc in all_docs:
            doc_type = doc["type"]
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        print(f"\nDocument type breakdown:")
        for doc_type, count in doc_types.items():
            print(f"- {doc_type}: {count}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()