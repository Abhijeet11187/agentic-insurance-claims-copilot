from customer_support_agent.core.settings import get_settings

s = get_settings()
print("app        :", s.app_name)
print("model      :", s.openai_model)
print("db path    :", s.db_path)
print("semantic   :", s.semantic_enabled)