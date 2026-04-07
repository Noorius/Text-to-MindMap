def create_outline(tree, root_name=None):
    '''
    tree: A tree resulted from recursive_tree_build function
    root_name: None or book_id
    '''
    lines = []

    if root_name:
        summary_text = root_name
    else:
        summary_text = tree['summary'].strip().replace('\n', '')
        if len(summary_text.split("summary:")) > 1:
            summary_text = summary_text.split("summary:")[1].replace("summary:", '').strip()
            
    lines.append(f"{summary_text}")

    if 'children' in tree and tree['children']:
        
        def recursive_extract(node_list, level=1):
            inner_lines = []
            indent = '\t' * level
            for item in node_list:
                if item.get('leaf', False):
                    summary = item['outline'].strip().replace('\n', '')
                else:
                    summary = item.get('summary', '').strip().replace('\n', '')

                if len(summary.split("summary:")) > 1:
                    summary = summary.split("summary:")[1].replace("summary:", '').strip()
        
                #print(item.get('leaf', False), summary)
                
                if summary:
                    inner_lines.append(indent + summary.rstrip('\n'))

                    if 'children' in item and item['children']:
                        il = recursive_extract(item['children'], level + 1)

                        inner_lines.extend(il)
                elif 'children' in item and item['children']:
                    il = recursive_extract(item['children'], level)
                    inner_lines.extend(il)

            return inner_lines

        il = recursive_extract(tree['children'], level=1)
        lines.extend(il)
    else:
        return '\n'.join(lines)

    return '\n'.join(lines)

def create_outline_pruned(tree, root_name=None, selected_events=[]):
    '''
    tree: A tree resulted from recursive_tree_build function
    root_name: None or book_id
    selected_events: selected events from rank_nodes function
    '''
    lines = []

    # Process the root node
    if root_name:
        summary_text = root_name
    else:
        summary_text = tree['summary'].strip().replace('\n', '')
        if "summary:" in summary_text:
            summary_text = summary_text.split("summary:")[1].strip()
            
    lines.append(f"{summary_text}")

    if 'children' in tree and tree['children']:
        
        def recursive_extract(node_list, level=1):
            inner_lines = []
            branch_has_selected = False  
            indent = '\t' * level
            
            for item in node_list:
                raw_summary = None
                
                # Extract summary based on whether it is a leaf or not
                if item.get('leaf', False):
                    summary = item['outline'].strip().replace('\n', '')
                    if len(summary.split("summary:")) > 1:
                        summary = summary.split("summary:")[1].strip()
                    # For a leaf, its raw comparison string comes from 'summary'
                    raw_summary = item.get('summary', None)
                else:
                    summary = item.get('summary', '').strip().replace('\n', '')
                    raw_summary = item.get('summary', None)
                
                # 1. Evaluate if THIS exact node's text is in selected_events
                is_selected = (raw_summary in selected_events) if raw_summary else False
                
                # 2. Look ahead to children to see if they are selected
                child_lines = []
                child_has_selected = False
                if 'children' in item and item['children']:
                    next_level = level + 1 if summary else level
                    child_lines, child_has_selected = recursive_extract(item['children'], next_level)
                
                # 3. Decision: Append this node ONLY if it is selected, OR if one of its children is selected
                if is_selected or child_has_selected:
                    branch_has_selected = True
                    if summary:
                        inner_lines.append(indent + summary.rstrip('\n'))
                    if child_lines:
                        inner_lines.extend(child_lines)

            return inner_lines, branch_has_selected

        # Execute recursion starting at depth 1
        il = recursive_extract(tree['children'], level=1)[0]
        lines.extend(il)