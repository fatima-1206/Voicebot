import json
import logging
from typing import Dict, List, Any, Optional, Set
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from gql.transport.exceptions import TransportError
import argparse
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphQLExtractor:
    def __init__(self, api_url: str, headers: Optional[Dict[str, str]] = None):
        """
        Initialize the GraphQL extractor
        
        Args:
            api_url: The GraphQL endpoint URL
            headers: Optional headers for authentication (e.g., {"Authorization": "Bearer token"})
        """
        self.api_url = api_url
        self.headers = headers or {}
        self.transport = RequestsHTTPTransport(
            url=api_url, 
            headers=self.headers,
            verify=True,
            retries=3
        )
        self.client = Client(transport=self.transport, fetch_schema_from_transport=True)
        self.schema = None
        
    def fetch_schema(self):
        """Fetch the GraphQL schema introspection"""
        try:
            introspection_query = gql("""
            query IntrospectionQuery {
              __schema {
                queryType { name }
                mutationType { name }
                subscriptionType { name }
                types {
                  ...FullType
                }
              }
            }

            fragment FullType on __Type {
              kind
              name
              description
              fields(includeDeprecated: true) {
                name
                description
                args {
                  ...InputValue
                }
                type {
                  ...TypeRef
                }
                isDeprecated
                deprecationReason
              }
              inputFields {
                ...InputValue
              }
              interfaces {
                ...TypeRef
              }
              enumValues(includeDeprecated: true) {
                name
                description
                isDeprecated
                deprecationReason
              }
              possibleTypes {
                ...TypeRef
              }
            }

            fragment InputValue on __InputValue {
              name
              description
              type { ...TypeRef }
              defaultValue
            }

            fragment TypeRef on __Type {
              kind
              name
              ofType {
                kind
                name
                ofType {
                  kind
                  name
                  ofType {
                    kind
                    name
                    ofType {
                      kind
                      name
                      ofType {
                        kind
                        name
                        ofType {
                          kind
                          name
                          ofType {
                            kind
                            name
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            """)
            
            # Use client.execute instead of client.fetch_schema
            self.schema = self.client.execute(introspection_query)
            logger.info("Schema fetched successfully")
            return self.schema
            
        except Exception as e:
            logger.error(f"Failed to fetch schema: {e}")
            raise

    def get_type_by_name(self, name: str) -> Optional[Dict]:
        """Get a type definition by name from the schema"""
        if not self.schema or not self.schema.get('__schema') or not self.schema['__schema'].get('types'):
            return None
        
        for type_def in self.schema['__schema']['types']:
            if type_def and type_def.get('name') == name:
                return type_def
        return None

    def _unwrap_type(self, type_ref: Dict) -> Optional[Dict]:
        """Unwrap NON_NULL and LIST types to get the actual type"""
        if not type_ref:
            return None
            
        if type_ref.get('kind') in ['NON_NULL', 'LIST']:
            of_type = type_ref.get('ofType')
            if of_type:
                return self._unwrap_type(of_type)
        return type_ref

    def _is_scalar_or_enum(self, type_name: str) -> bool:
        """Check if a type is a scalar or enum type"""
        if type_name in ['String', 'Int', 'Float', 'Boolean', 'ID']:
            return True
        
        type_def = self.get_type_by_name(type_name)
        if type_def:
            return type_def.get('kind') in ['SCALAR', 'ENUM']
        return False

    def build_query_fields(self, type_name: str, max_depth: int = 3, current_depth: int = 0, visited_types: Optional[Set[str]] = None) -> str:
        """
        Build query fields for a given type
        
        Args:
            type_name: Name of the GraphQL type
            max_depth: Maximum depth for nested queries
            current_depth: Current recursion depth
            visited_types: Set of types already visited to prevent infinite recursion
        """
        if visited_types is None:
            visited_types = set()
            
        if current_depth >= max_depth or type_name in visited_types:
            return ""
            
        # Handle scalar and enum types
        if self._is_scalar_or_enum(type_name):
            return ""
            
        # Get type definition
        type_def = self.get_type_by_name(type_name)
        if not type_def or type_def.get('kind') not in ['OBJECT', 'INTERFACE']:
            return ""
            
        visited_types.add(type_name)
        
        fields = type_def.get('fields', [])
        if not fields:
            visited_types.discard(type_name)
            return ""
            
        field_selections = []
        
        # Always include __typename for debugging
        field_selections.append("__typename")
        
        # Process each field
        for field in fields:
            if not field or not field.get('name'):
                continue
                
            field_name = field['name']
            
            # Skip fields that require arguments
            field_args = field.get('args', [])
            has_required_args = any(
                arg and arg.get('type', {}).get('kind') == 'NON_NULL' 
                for arg in field_args
            )
            
            if has_required_args:
                continue
                
            # Get the field's type
            field_type = field.get('type')
            if not field_type:
                continue
                
            actual_type = self._unwrap_type(field_type)
            if not actual_type:
                continue
                
            actual_type_name = actual_type.get('name')
            if not actual_type_name:
                continue
                
            # If it's a scalar/enum, just add the field name
            if self._is_scalar_or_enum(actual_type_name):
                field_selections.append(field_name)
            else:
                # It's an object type, need to build nested fields
                nested_fields = self.build_query_fields(
                    actual_type_name, 
                    max_depth, 
                    current_depth + 1, 
                    visited_types.copy()
                )
                
                if nested_fields:
                    field_selections.append(f"{field_name} {{ {nested_fields} }}")
                # If we can't build nested fields, skip this field entirely
                # (don't add just the field name as that will cause the error)
        
        visited_types.discard(type_name)
        return " ".join(field_selections)

    def build_query_for_field(self, field: Dict, max_depth: int = 3) -> str:
        """
        Build a complete GraphQL query for a root field
        
        Args:
            field: Root field definition from schema
            max_depth: Maximum depth for nested queries
        """
        if not field or not field.get('name'):
            return ""
            
        field_name = field['name']
        field_type = field.get('type')
        
        if not field_type:
            return field_name
            
        # Unwrap the type
        actual_type = self._unwrap_type(field_type)
        if not actual_type:
            return field_name
            
        actual_type_name = actual_type.get('name')
        if not actual_type_name:
            return field_name
            
        # If it's a scalar/enum, just return the field name
        if self._is_scalar_or_enum(actual_type_name):
            return field_name
            
        # Build nested fields for object types
        nested_fields = self.build_query_fields(actual_type_name, max_depth)
        
        if nested_fields:
            return f"{field_name} {{ {nested_fields} }}"
        else:
            return field_name

    def extract_all_data(self, max_depth: int = 3, batch_size: int = 3) -> Dict[str, Any]:
        """
        Extract all available data from the GraphQL API
        
        Args:
            max_depth: Maximum depth for nested queries
            batch_size: Number of fields to query at once (reduced default)
        """
        if not self.schema:
            self.fetch_schema()
            
        # Add proper null checks
        if not self.schema or not self.schema.get('__schema'):
            logger.error("Invalid schema structure")
            return {}
            
        query_type_info = self.schema['__schema'].get('queryType')
        if not query_type_info or not query_type_info.get('name'):
            logger.error("No query type found in schema")
            return {}
            
        query_type_name = query_type_info['name']
        query_type = self.get_type_by_name(query_type_name)
        
        if not query_type:
            logger.error(f"Query type '{query_type_name}' not found")
            return {}
            
        fields = query_type.get('fields', [])
        if not fields:
            logger.error("No query fields found")
            return {}
            
        all_data = {}
        
        logger.info(f"Found {len(fields)} root query fields")
        
        # Process fields individually to avoid errors affecting other fields
        for field in fields:
            if not field or not field.get('name'):
                continue
                
            field_name = field['name']
            logger.info(f"Processing field: {field_name}")
            
            try:
                # Handle fields with required arguments - improved detection
                field_args = field.get('args', [])
                required_args = []
                optional_args = []
                
                for arg in field_args:
                    if not arg or not arg.get('name'):
                        continue
                        
                    arg_type = arg.get('type', {})
                    if arg_type.get('kind') == 'NON_NULL':
                        required_args.append({
                            'name': arg.get('name', 'unknown'),
                            'type': self._get_type_name(arg_type)
                        })
                    else:
                        optional_args.append({
                            'name': arg.get('name', 'unknown'),
                            'type': self._get_type_name(arg_type)
                        })
                
                if required_args:
                    logger.warning(f"Skipping {field_name} - requires arguments: {[arg['name'] for arg in required_args]}")
                    all_data[field_name] = {
                        "skipped": "requires_arguments", 
                        "required_args": required_args,
                        "optional_args": optional_args
                    }
                    continue
                
                # Build and execute query for this field
                query_body = self.build_query_for_field(field, max_depth)
                if not query_body:
                    logger.warning(f"Could not build query for {field_name}")
                    all_data[field_name] = {"skipped": "could_not_build_query"}
                    continue
                    
                query_str = f"query {{ {query_body} }}"
                logger.info(f"Generated query: {query_str}")
                
                query = gql(query_str)
                result = self.client.execute(query)
                all_data[field_name] = result.get(field_name)
                
                logger.info(f"✓ Successfully extracted {field_name}")
                
            except TransportError as e:
                logger.error(f"Transport error for {field_name}: {e}")
                all_data[field_name] = {"error": "transport_error", "message": str(e)}
            except Exception as e:
                logger.error(f"Error extracting {field_name}: {e}")
                all_data[field_name] = {"error": "extraction_error", "message": str(e)}
                
        return all_data

    def _get_type_name(self, type_ref: Dict) -> str:
        """Get the name of a type, unwrapping NON_NULL and LIST wrappers"""
        if not type_ref:
            return "Unknown"
            
        if type_ref.get('name'):
            return type_ref['name']
            
        if type_ref.get('kind') in ['NON_NULL', 'LIST']:
            of_type = type_ref.get('ofType')
            if of_type:
                return self._get_type_name(of_type)
                
        return "Unknown"

    def save_to_json(self, data: Dict[str, Any], filename: str = "extracted_data.json"):
        """Save extracted data to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"Data saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise

    def generate_schema_summary(self) -> Dict[str, Any]:
        """Generate a summary of the GraphQL schema"""
        if not self.schema:
            self.fetch_schema()
            
        if not self.schema or not self.schema.get('__schema'):
            return {"error": "Invalid schema structure"}
            
        schema_info = self.schema['__schema']
        
        summary = {
            "query_type": schema_info.get('queryType', {}).get('name') if schema_info.get('queryType') else None,
            "mutation_type": schema_info.get('mutationType', {}).get('name') if schema_info.get('mutationType') else None,
            "subscription_type": schema_info.get('subscriptionType', {}).get('name') if schema_info.get('subscriptionType') else None,
            "types": {},
            "total_types": len(schema_info.get('types', []))
        }
        
        # Get query fields summary
        if summary["query_type"]:
            query_type = self.get_type_by_name(summary["query_type"])
            if query_type and query_type.get('fields'):
                summary["available_queries"] = []
                for field in query_type['fields']:
                    if not field or not field.get('name'):
                        continue
                        
                    field_info = {
                        "name": field['name'],
                        "description": field.get('description'),
                        "args": []
                    }
                    
                    field_args = field.get('args', [])
                    for arg in field_args:
                        if arg and arg.get('name'):
                            arg_info = {
                                "name": arg['name'],
                                "required": arg.get('type', {}).get('kind') == 'NON_NULL'
                            }
                            field_info["args"].append(arg_info)
                    
                    summary["available_queries"].append(field_info)
        
        types_list = schema_info.get('types', [])
        for type_def in types_list:
            if not type_def or not type_def.get('name') or type_def['name'].startswith('__'):
                continue  # Skip introspection types
                
            type_fields = type_def.get('fields', [])
            summary['types'][type_def['name']] = {
                "kind": type_def.get('kind'),
                "fields_count": len(type_fields) if type_fields else 0,
                "description": type_def.get('description')
            }
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Extract all data from a GraphQL API')
    parser.add_argument('url', help='GraphQL endpoint URL')
    parser.add_argument('--headers', help='JSON string of headers (e.g., \'{"Authorization": "Bearer token"}\')')
    parser.add_argument('--output', '-o', default='extracted_data.json', help='Output JSON file')
    parser.add_argument('--schema-only', action='store_true', help='Only extract schema summary')
    parser.add_argument('--max-depth', type=int, default=3, help='Maximum query depth')
    parser.add_argument('--batch-size', type=int, default=3, help='Batch size for field processing')
    
    args = parser.parse_args()
    
    # Parse headers if provided
    headers = {}
    if args.headers:
        try:
            headers = json.loads(args.headers)
        except json.JSONDecodeError:
            logger.error("Invalid JSON format for headers")
            sys.exit(1)
    
    try:
        extractor = GraphQLExtractor(args.url, headers)
        
        if args.schema_only:
            schema_summary = extractor.generate_schema_summary()
            extractor.save_to_json(schema_summary, args.output.replace('.json', '_schema.json'))
        else:
            # Extract all data
            logger.info("Starting data extraction...")
            data = extractor.extract_all_data(max_depth=args.max_depth, batch_size=args.batch_size)
            
            # Also save schema summary
            schema_summary = extractor.generate_schema_summary()
            extractor.save_to_json(schema_summary, args.output.replace('.json', '_schema.json'))
            
            # Save extracted data
            extractor.save_to_json(data, args.output)
            
            logger.info(f"Extraction complete! Data saved to {args.output}")
            logger.info(f"Schema summary saved to {args.output.replace('.json', '_schema.json')}")
            
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



#     Countries	Sample Dataset	https://countries.trevorblades.com/
# Rick & Morty	Public GraphQL API	https://rickandmortyapi.com/graphql