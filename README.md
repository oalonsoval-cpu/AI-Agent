# AI-Agent
Desarrollo de un agente de inteligencia artificial en Python utilizando frameworks modernos como 'LangChain' y modelos de lenguaje avanzados como Claude y GPT.

**Aspectos clave:**
- Integración y utilización de distintos LLMs (gestión de claves API).
- Estructuración de la salida para su uso en aplicaciónes.
- Creación de plantillas de prompt para mejorar la interacción.
- Desarrollo y ejecución completa del agente.
- Procesamiento y análisis de la salida.
- Integración de herramientas preexistentes (DuckDuckGo, Wikipedia).
- Ampliación de capacidades mediante herramientas personalizadas.

---

# Desarrollo

## 1.- Resolviendo dependencias
1. Creación del fichero `requirements.txt:`

```
langchain
wikipedia
langchain-community
langchain-openai
langchain-anthropic
python-dotenv
pydantic
duckduckgo-search
```

- `langchain:` Framework para construir aplicaciones de lenguaje natural (LLMs).`
- `wikipedia:` Cliente de Python para acceder a Wikipedia.
- `langchain-community:` Extensiones y contribuciones de la comunidad de LangChain.
- `langchain-openai:` Adaptador específico de LangChain para usar los modelos de OpenAI.
- `langchain-antrophic:` Similar a langchain-openai, pero para los modelos de Anthropic (p. ej., Claude).
- `python-dotenv:` Permite cargar variables de entorno desde archivos .env en Python. Muy útil para ocultar claves de API, configuraciones y secretos sin ponerlos en el código.
- `pydantic:` Permite definir clases con tipos de datos, validarlos automáticamente y convertir datos entrantes (por ejemplo JSON) en objetos Python.
- `duckduckgo-search:` Cliente Python para realizar búsquedas en DuckDuckGo.

2. Creación y activación del entorno virtual para aislar dependencias.
`python -m venv venv`
`source ./venv/bin/activate`

3.- Instalación de dependencias
`pip install -m requirements.txt`

## 2.- Fichero `main.py`
1. Configuración del entorno y librerías
```
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor

from tools import search_tool, wiki_tool, save_tool

load_dotenv()
```

- `ChatPromptTemplate:` Define la plantilla de interacción con el modelo de lenguaje.
Contiene los mensajes del sistema, del usuario y del historial, que guían cómo el LLM debe responder.
- `PydanticOutputParser:` Convierte la salida libre del LLM en un objeto estructurado de Python. Garantiza que la respuesta cumpla con el modelo de datos definido por Pydantic, incluyendo: `topic` (tema investigado), `summary` (resumen generado), `sources` (referencias consultadas) y `tools_used` (herramientas utilizadas).
- `AgentExecutor:` Ejecuta el agente LangChain con las herramientas y el prompt configurados.
Maneja la interacción entre el usuario, el LLM y las herramientas externas.

2. Definición del modelo de respuesta estructurada
```
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
```
Clase que respresenta la estructura de la respuesta según el modelo de datos definido por Pydantic.

3. Inicialización del LLM y prueba rápida
```
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("What is the meaning of life?")
print(response)
```
Crea una instancia del modelo de lenguaje de OpenAI (GPT-4o-mini) usando LangChain y le pasa un prompt de ejemplo.

4. Creación del prompt y parser
```
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a researcher assistant ..."),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())
```
- **Parser:** define la estructura de la respuesta.
- **PromptTemplate:** define cómo debe responder el LLM.
- **`partial(format_instructions)`:** asegura que la respuesta del LLM encaje con la estructura del parser.

5. Definición del agente y ejecución
```
tools = [search_tool, wiki_tool, save_tool]
agent = create_agent(llm=llm, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
query = input("What can I help you research?")
raw_response = agent_executor.invoke({"query": query})
```

`AgentExecutor` es el controlador que ejecuta el agente:
- Recibe la consulta del usuario (`query`).
- El agente decide qué herramientas usar y cómo generar la respuesta.
- Devuelve la respuesta cruda (`raw_response`) para ser procesada o parseada.
- `verbose=True` permite ver pasos intermedios durante la ejecución, útil para depuración.

6. Parseo y visualización de la respuesta estructurada
```
try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
```

## 3.- Fichero `tools.py`
1. Tool para guardar resultados
```
@tool
def save_tool(data: str, filename: str = "research_output.txt"):
    """Saves structured research data to a text file."""
```
- Guarda los resultados de investigación en un archivo .txt.
- Añade un timestamp para mantener un historial de consultas.
- Permite conservar la información de manera organizada y reutilizable.

2. Tool para búsqueda web
```
search = DuckDuckGoSearchRun()
@tool
def search_tool(query:str):
    """Search the web information."""
    return search.run(query)

```
- Realiza búsquedas en DuckDuckGo.
- Permite al agente recuperar información actualizada de la web.

3. Tool para Wikipedia
```
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
```
- Consulta artículos de Wikipedia y devuelve resúmenes concisos.
- `top_k_results=1` garantiza que solo se use el artículo más relevante.
- `doc_content_chars_max=100` limita la cantidad de texto para que la respuesta sea breve y manejable.
