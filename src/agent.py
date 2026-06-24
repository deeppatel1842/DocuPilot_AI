import os
from datetime import datetime
from aisuite import Client
from .research_tools import (
    arxiv_search_tool,
    tavily_search_tool,
    wikipedia_search_tool,
)
from .trace_logger import traced_completion, wrap_tool

client = Client()

# Models are configurable via .env. Defaults favour speed/cost: a fast planner
# and editor, with the quality-critical writer kept on gpt-4.1-mini.
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "openai:gpt-4.1-mini")
WRITER_MODEL = os.getenv("WRITER_MODEL", "openai:gpt-4.1-mini")
# gpt-4.1-nano was tried for the editor but it corrupted arXiv IDs (citation
# coverage dropped to 0%), so the default editor stays on gpt-4.1-mini.
EDITOR_MODEL = os.getenv("EDITOR_MODEL", "openai:gpt-4.1-mini")


# === Research Agent ===
def research_agent(
    prompt: str, model: str = RESEARCH_MODEL, return_messages: bool = False
):
    print("==================================")
    print("🔍 Research Agent")
    print("==================================")

    full_prompt = f"""
You are a research assistant. Gather accurate, relevant, well-sourced information for the
request below, then summarise concisely.

Tools (use what fits; more than one is fine):
- tavily_search_tool: general web search (recent news, blogs, industry, practical info).
- arxiv_search_tool: academic papers. ONLY for Computer Science, Mathematics, Physics,
  Statistics, EE/Systems Science, Economics, Quantitative Biology/Finance. Never other domains.
- wikipedia_search_tool: background, definitions, overviews.

Rules: choose focused queries; verify across sources when possible; never fabricate sources.

Return a structured findings summary:
1. Approach - tools used and why.
2. Key findings - grouped, each with source attribution.
3. Sources - title, URL, date/author when available.
4. Limitations - any gaps.

Today is {datetime.now().strftime("%Y-%m-%d")}.

RESEARCH REQUEST:
{prompt}
""".strip()

    messages = [{"role": "user", "content": full_prompt}]
    tools = [
        wrap_tool(arxiv_search_tool),
        wrap_tool(tavily_search_tool),
        wrap_tool(wikipedia_search_tool),
    ]

    try:
        resp = traced_completion(
            client,
            agent="research_agent",
            phase="research",
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_turns=3,
            temperature=0.0,  # Use deterministic output
        )

        content = resp.choices[0].message.content or ""

        # ---- Collect tool calls from intermediate_responses and intermediate_messages
        calls = []

        # A) From intermediate_responses
        for ir in getattr(resp, "intermediate_responses", []) or []:
            try:
                tcs = ir.choices[0].message.tool_calls or []
                for tc in tcs:
                    calls.append((tc.function.name, tc.function.arguments))
            except Exception:
                pass

        # B) From intermediate_messages on the final message
        for msg in getattr(resp.choices[0].message, "intermediate_messages", []) or []:
            # assistant message with tool_calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    calls.append((tc.function.name, tc.function.arguments))

        # Dedup while preserving order
        seen = set()
        dedup_calls = []
        for name, args in calls:
            key = (name, args)
            if key not in seen:
                seen.add(key)
                dedup_calls.append((name, args))

        # Pretty print args: JSON->dict if possible
        tool_lines = []
        for name, args in dedup_calls:
            arg_text = str(args)
            try:
                import json as _json

                parsed = _json.loads(args) if isinstance(args, str) else args
                if isinstance(parsed, dict):
                    kv = ", ".join(f"{k}={repr(v)}" for k, v in parsed.items())
                    arg_text = kv
            except Exception:
                # keep raw string if not JSON
                pass
            tool_lines.append(f"- {name}({arg_text})")

        if tool_lines:
            tools_html = (
                "<h2 style='font-size:1.5em; color:#2563eb;'>📎 Tools used</h2>"
            )
            tools_html += (
                "<ul>" + "".join(f"<li>{line}</li>" for line in tool_lines) + "</ul>"
            )
            content += "\n\n" + tools_html

        print("✅ Output:\n", content)
        return content, messages

    except Exception as e:
        print("❌ Error:", e)
        return f"[Model Error: {str(e)}]", messages


def writer_agent(
    prompt: str,
    model: str = WRITER_MODEL,
    min_words_total: int = 2400,
    min_words_per_section: int = 400,
    max_tokens: int = 15000,
    retries: int = 1,
):
    print("==================================")
    print("✍️ Writer Agent")
    print("==================================")

    system_message = """
You are an expert academic writer with a PhD-level understanding of scholarly communication. Your task is to synthesize research materials into a comprehensive, well-structured academic report.

GROUND EVERYTHING IN THE PROVIDED SOURCES. Do not rely on prior knowledge for facts or
citations; use only what the research context provides, and cite it explicitly.

## REPORT REQUIREMENTS:
- Produce a COMPLETE, POLISHED, and PUBLICATION-READY academic report in Markdown format
- Create original content that thoroughly analyzes the provided research materials
- DO NOT merely summarize the sources; develop a cohesive narrative with critical analysis
- Length should be appropriate to thoroughly cover the topic (typically 1500-3000 words)

## MANDATORY STRUCTURE:
1. **Title**: Clear, concise, and descriptive of the content
2. **Abstract**: Brief summary (100-150 words) of the report's purpose, methods, and key findings
3. **Introduction**: Present the topic, research question/problem, significance, and outline of the report
4. **Background/Literature Review**: Contextualize the topic within existing scholarship
5. **Methodology**: If applicable, describe research methods, data collection, and analytical approaches
6. **Key Findings/Results**: Present the primary outcomes and evidence
7. **Discussion**: Interpret findings, address implications, limitations, and connections to broader field
8. **Conclusion**: Synthesize main points and suggest directions for future research
9. **References**: Complete list of all cited works

## ACADEMIC WRITING GUIDELINES:
- Maintain formal, precise, and objective language throughout
- Use discipline-appropriate terminology and concepts
- Support all claims with evidence and reasoning
- Develop logical flow between ideas, paragraphs, and sections
- Include relevant examples, case studies, data, or equations to strengthen arguments
- Address potential counterarguments and limitations

## CITATION AND REFERENCE RULES (STRICT - these are graded):
- Use ONLY sources that appear in the provided research findings/context. Do NOT add
  papers, URLs or references from your own memory, and NEVER write placeholder or
  "hypothetical" citations.
- Copy each source URL VERBATIM from the research context into the References section.
  Do not normalise, shorten, invent, or swap arXiv IDs/URLs.
- Every section must contain inline numeric citations [1], [2], ... grounded in those
  sources. Cite the MAJORITY of the provided sources at least once.
- Each inline [n] must map to exactly one References entry, and every References entry
  must be cited at least once in the text.
- If a claim is not supported by a provided source, either remove it or clearly mark it
  as general background (no citation) - do not fabricate support.
- Preserve all original titles, authors, dates, URLs and DOIs from the source materials.

## FORMATTING GUIDELINES:
- Use Markdown syntax for all formatting (headings, emphasis, lists, etc.)
- Include appropriate section headings and subheadings to organize content
- Format any equations, tables, or figures according to academic conventions
- Use bullet points or numbered lists when appropriate for clarity
- Use html syntax to handle all links with target="_blank", so user can always open link in new tab on both html and markdown format

Output the complete report in Markdown format only. Do not include meta-commentary about the writing process.

INTERNAL CHECKLIST (DO NOT INCLUDE IN OUTPUT):
- [ ] Incorporated all provided research materials
- [ ] Developed original analysis beyond mere summarization
- [ ] Included all mandatory sections with appropriate content
- [ ] Used proper inline citations for all borrowed content
- [ ] Created complete References section with all cited sources
- [ ] Maintained academic tone and language throughout
- [ ] Ensured logical flow and coherent structure
- [ ] Preserved all source URLs and bibliographic information
""".strip()

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    def _call(messages_):
        resp = traced_completion(
            client,
            agent="writer_agent",
            phase="writer",
            model=model,
            messages=messages_,
            temperature=0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _word_count(md_text: str) -> int:
        import re

        words = re.findall(r"\b\w+\b", md_text)
        return len(words)

    content = _call(messages)

    print("✅ Output:\n", content)
    return content, messages


def editor_agent(
    prompt: str,
    model: str = EDITOR_MODEL,
    target_min_words: int = 2400,
):
    print("==================================")
    print("🧠 Editor Agent")
    print("==================================")

    system_message = """
You are a professional academic editor with expertise in improving scholarly writing across disciplines. Your task is to refine and elevate the quality of the academic text provided.

## Your Editing Process:
1. Analyze the overall structure, argument flow, and coherence of the text
2. Ensure logical progression of ideas with clear topic sentences and transitions between paragraphs
3. Improve clarity, precision, and conciseness of language while maintaining academic tone
4. Verify technical accuracy (to the extent possible based on context)
5. Enhance readability through appropriate formatting and organization

## Specific Elements to Address:
- Strengthen thesis statements and main arguments
- Clarify complex concepts with additional explanations or examples where needed
- Add relevant equations, diagrams, or illustrations (described in markdown) when they would enhance understanding
- Ensure proper integration of evidence and maintain academic rigor
- Standardize terminology and eliminate redundancies
- Improve sentence variety and paragraph structure
- Preserve every inline citation [1], [2], ... and keep the References section intact.
  Do NOT remove citations, invent new sources, or alter source URLs. Where the draft has
  uncited claims, add inline citations to the existing provided sources where possible.

## Formatting Guidelines:
- Use markdown formatting consistently for headings, emphasis, lists, etc.
- Structure content with appropriate section headings and subheadings
- Format equations, tables, and figures according to academic standards

Return only the revised, polished text in Markdown format without explanatory comments about your edits.
""".strip()

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    response = traced_completion(
        client,
        agent="editor_agent",
        phase="editor",
        model=model,
        messages=messages,
        temperature=0,
    )

    content = response.choices[0].message.content
    print("✅ Output:\n", content)
    return content, messages