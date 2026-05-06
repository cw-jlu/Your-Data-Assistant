import re
import json
import asyncio
from pathlib import Path
from typing import Any, List, Dict, Optional

def extract_nodes_from_markdown(markdown_content: str, max_chunk_lines: int = 150):
    header_pattern = r'^(#{1,6})\s+(.+)$'
    code_block_pattern = r'^```'
    
    lines = markdown_content.split('\n')
    in_code_block = False
    
    nodes = []
    current_node = None
    
    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()
        
        if re.match(code_block_pattern, stripped_line):
            in_code_block = not in_code_block
            
        if not in_code_block and re.match(header_pattern, stripped_line):
            match = re.match(header_pattern, stripped_line)
            level = len(match.group(1))
            title = match.group(2).strip()
            
            if current_node:
                current_node['end_line'] = line_num - 1
                nodes.append(current_node)
                
            current_node = {
                'title': title,
                'line_num': line_num,
                'level': level,
                'is_header': True
            }
        elif current_node is None and stripped_line:
            current_node = {
                'title': 'Document Start / Prologue',
                'line_num': line_num,
                'level': 1,
                'is_header': False
            }
            
    if current_node:
        current_node['end_line'] = len(lines)
        nodes.append(current_node)
        
    if not nodes:
        return [], lines
        
    final_nodes = []
    for node in nodes:
        node_len = node['end_line'] - node['line_num'] + 1
        if node_len > max_chunk_lines * 2:
            if node['is_header']:
                final_nodes.append(node)
            
            start_line = node['line_num'] + 1 if node['is_header'] else node['line_num']
            while start_line <= node['end_line']:
                expected_end = min(start_line + max_chunk_lines - 1, node['end_line'])
                actual_end = expected_end
                
                # Search for a clean paragraph break (empty line) near the expected end
                if expected_end < node['end_line']:
                    found_break = False
                    # Look backward up to 50 lines
                    for offset in range(0, min(50, expected_end - start_line)):
                        if not lines[expected_end - 1 - offset].strip():
                            actual_end = expected_end - offset
                            found_break = True
                            break
                    # If not found, look forward up to 50 lines
                    if not found_break:
                        for offset in range(1, min(50, node['end_line'] - expected_end)):
                            if not lines[expected_end - 1 + offset].strip():
                                actual_end = expected_end + offset
                                break
                                
                title_prefix = "Content Section" if node['is_header'] else "Text Chunk"
                child_level = node['level'] + 1 if node['is_header'] else node['level']
                final_nodes.append({
                    'title': f"{title_prefix} ({start_line}-{actual_end})",
                    'line_num': start_line,
                    'end_line': actual_end,
                    'level': child_level,
                    'is_header': False
                })
                start_line = actual_end + 1
        else:
            final_nodes.append(node)
            
    for node in final_nodes:
        node['text'] = '\n'.join(lines[node['line_num']-1 : node['end_line']]).strip()
        
    return final_nodes, lines

def extract_node_text_content(node_list: List[Dict], markdown_lines: List[str]):
    # node_list already has 'text', 'line_num', 'end_line', 'level' populated in the new logic.
    # This is kept for compatibility but doesn't need to re-read.
    return node_list

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
            'text': node.get('text', ''),
            'line_num': node['line_num'],
            'end_line': node['end_line'],
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

async def generate_summaries_recursively_async(nodes: List[Dict], model_adapter: Any, max_summaries: int = 1000):
    all_nodes = []
    def _collect(node_list):
        for node in node_list:
            all_nodes.append(node)
            if 'nodes' in node:
                _collect(node['nodes'])
    
    _collect(nodes)
    
    # Generate summaries for all nodes that have text
    candidates = [n for n in all_nodes if n.get('text', '').strip()]
    
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
        lines.append('  ' * indent + f"[{node['node_id']}] {node['title']} (lines {node['line_num']}-{node['end_line']}){summary_str}")
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
