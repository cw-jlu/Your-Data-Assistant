import re
import json
import asyncio
from pathlib import Path
from typing import Any, List, Dict, Optional

def extract_nodes_from_markdown(markdown_content: str):
    header_pattern = r'^(#{1,6})\s+(.+)$'
    code_block_pattern = r'^```'
    node_list = []
    
    lines = markdown_content.split('\n')
    in_code_block = False
    
    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()
        
        if re.match(code_block_pattern, stripped_line):
            in_code_block = not in_code_block
            continue
        
        if not stripped_line:
            continue
        
        if not in_code_block:
            match = re.match(header_pattern, stripped_line)
            if match:
                title = match.group(2).strip()
                node_list.append({'node_title': title, 'line_num': line_num})

    return node_list, lines

def extract_node_text_content(node_list: List[Dict], markdown_lines: List[str]):
    all_nodes = []
    for node in node_list:
        line_content = markdown_lines[node['line_num'] - 1]
        header_match = re.match(r'^(#{1,6})', line_content)
        
        if header_match is None:
            continue
            
        processed_node = {
            'title': node['node_title'],
            'line_num': node['line_num'],
            'level': len(header_match.group(1))
        }
        all_nodes.append(processed_node)
    
    for i, node in enumerate(all_nodes):
        start_line = node['line_num'] - 1 
        if i + 1 < len(all_nodes):
            end_line = all_nodes[i + 1]['line_num'] - 1 
        else:
            end_line = len(markdown_lines)
        
        node['text'] = '\n'.join(markdown_lines[start_line:end_line]).strip()    
    return all_nodes

def build_tree_from_nodes(node_list: List[Dict]):
    if not node_list:
        return []
    
    stack = []
    root_nodes = []
    node_counter = 1
    
    for node in node_list:
        current_level = node['level']
        
        tree_node = {
            'title': node['title'],
            'node_id': str(node_counter).zfill(4),
            'text': node['text'],
            'line_num': node['line_num'],
            'nodes': []
        }
        node_counter += 1
        
        while stack and stack[-1][1] >= current_level:
            stack.pop()
        
        if not stack:
            root_nodes.append(tree_node)
        else:
            parent_node, parent_level = stack[-1]
            parent_node['nodes'].append(tree_node)
        
        stack.append((tree_node, current_level))
    
    return root_nodes

async def generate_node_summary_async(node: Dict, model_adapter: Any) -> str:
    from data_agent_baseline.agents.model import ModelMessage
    # Use only the first 2000 chars for summary to be fast
    prompt = f"Summarize the following document section in one very short sentence (max 15 words). Focus on key entities.\n\nSection Content:\n{node['text'][:2000]}"
    try:
        # Wrap sync call in thread
        response = await asyncio.to_thread(
            model_adapter.complete, 
            [ModelMessage(role="user", content=prompt)]
        )
        return response.strip()
    except Exception:
        return ""

async def generate_summaries_recursively_async(nodes: List[Dict], model_adapter: Any, max_summaries: int = 15):
    all_nodes = []
    def _collect(node_list):
        for node in node_list:
            all_nodes.append(node)
            if 'nodes' in node:
                _collect(node['nodes'])
    
    _collect(nodes)
    
    # Filter for nodes that are worth summarizing
    candidates = [n for n in all_nodes if len(n['text']) > 400]
    # Sort by length to prioritize large sections, but limit total
    candidates = sorted(candidates, key=lambda x: len(x['text']), reverse=True)[:max_summaries]
    
    if not candidates:
        return nodes
    
    tasks = [generate_node_summary_async(node, model_adapter) for node in candidates]
    summaries = await asyncio.gather(*tasks)
    
    for node, summary in zip(candidates, summaries):
        if summary:
            node['summary'] = summary
            
    return nodes

def format_tree_for_roadmap(tree: List[Dict], indent: int = 0) -> List[str]:
    lines = []
    for node in tree:
        summary = node.get('summary', '')
        summary_str = f" - {summary}" if summary else ""
        lines.append('  ' * indent + f"[{node['node_id']}] {node['title']} (line {node['line_num']}){summary_str}")
        if 'nodes' in node and node['nodes']:
            lines.extend(format_tree_for_roadmap(node['nodes'], indent + 1))
    return lines

async def get_md_pageindex_summary_async(md_path: Path, model_adapter: Any = None) -> str:
    try:
        content = md_path.read_text(encoding="utf-8", errors="replace")
        node_list, lines = extract_nodes_from_markdown(content)
        nodes_with_content = extract_node_text_content(node_list, lines)
        tree = build_tree_from_nodes(nodes_with_content)
        
        if model_adapter:
            await generate_summaries_recursively_async(tree, model_adapter)
        
        roadmap_lines = format_tree_for_roadmap(tree)
        return "\n".join(roadmap_lines)
    except Exception as e:
        return f"Error indexing {md_path.name}: {e}"

def get_pageindex_roadmap(context_dir: Path, model_adapter: Any = None) -> str:
    md_files = sorted(context_dir.rglob("*.md"))
    md_files = [f for f in md_files if f.name.lower() != "knowledge.md"]
    
    if not md_files:
        return ""
    
    # We need to run the async parts
    async def _build():
        tasks = [get_md_pageindex_summary_async(f, model_adapter) for f in md_files]
        summaries = await asyncio.gather(*tasks)
        
        roadmap_parts = ["=== PAGEINDEX TREE STRUCTURES ==="]
        for md_file, summary in zip(md_files, summaries):
            rel_path = md_file.relative_to(context_dir)
            roadmap_parts.append(f"\nDocument: {rel_path}")
            roadmap_parts.append(summary)
        return "\n".join(roadmap_parts)
    
    try:
        # Check if we are already in an event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # This is tricky in a sync environment. 
            # Since ReActAgent.run is sync, we'll use a new thread + loop if needed, 
            # or just run it via asyncio.run if not in a loop.
            import threading
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(asyncio.run, _build()).result()
        else:
            return asyncio.run(_build())
    except RuntimeError:
        return asyncio.run(_build())
