# Backend DTO Adjustment: `pydantic_class_json_schema`

`query.output_format.pydantic_class_json_schema` is sent as a **JSON string**
(not a nested object) for canonical key ordering in the cache layer.

```json
{
  "pydantic_class_json_schema": "{\"properties\": {\"title\": ...}}"
}
```

The backend should `JSON.parse()` this field before storing or processing it.

# Files API Note

`/files/update/:id` could be renamed to just `POST /files/:id` or
`PATCH /files/:id` to be more RESTful.
