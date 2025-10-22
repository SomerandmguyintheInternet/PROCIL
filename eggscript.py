"""
Eggscript Language Engine - Enhanced Version
- Core execution engine for .egg and .eggless files
- Handles variable substitution, control flow, functions, and safe evaluation
- Implements comprehensive language features with security
"""
import re
import json
import base64
import random
import string
import ast
import operator
from pathlib import Path
from typing import Any, Dict, List, Optional

class GRUMError(Exception):
    """Base exception for Eggscript runtime errors"""
    def __init__(self, message, line_number=None):
        self.line_number = line_number
        super().__init__(f"[GRUM] {message}" + (f" (line {line_number})" if line_number else ""))

def obfuscate(text):
    """Base64 encode text for .egg files"""
    return base64.b64encode(text.encode()).decode()

def deobfuscate(text):
    """Base64 decode text from .egg files"""
    try:
        return base64.b64decode(text.encode()).decode()
    except Exception:
        raise GRUMError("Invalid obfuscated .egg file")

class SafeEvaluator:
    """Safe expression evaluator using AST whitelist"""
    
    ALLOWED_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare,
        ast.Constant, ast.Num, ast.Str, ast.NameConstant,
        ast.Name, ast.Load, ast.List, ast.Tuple, ast.Dict,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq,
        ast.And, ast.Or, ast.Not, ast.UAdd, ast.USub,
        ast.Subscript, ast.Index, ast.Slice,
        ast.Call, ast.keyword, ast.Attribute
    }
    
    SAFE_BUILTINS = {
        'str': str, 'int': int, 'float': float, 'bool': bool,
        'len': len, 'range': range, 'min': min, 'max': max,
        'sum': sum, 'abs': abs, 'round': round,
        'dict': dict, 'list': list, 'set': set, 'tuple': tuple,
        'sorted': sorted, 'reversed': reversed, 'enumerate': enumerate,
        'zip': zip, 'map': map, 'filter': filter,
        'any': any, 'all': all,
    }
    
    def __init__(self, extra_context=None):
        self.context = {**self.SAFE_BUILTINS}
        if extra_context:
            self.context.update(extra_context)
    
    def is_safe(self, node):
        """Check if AST node is in whitelist"""
        if type(node) not in self.ALLOWED_NODES:
            return False
        for child in ast.walk(node):
            if type(child) not in self.ALLOWED_NODES:
                return False
        return True
    
    def evaluate(self, expression: str, local_vars: Dict[str, Any] = None) -> Any:
        """Safely evaluate expression"""
        try:
            tree = ast.parse(expression, mode='eval')
            if not self.is_safe(tree):
                raise GRUMError(f"Disallowed syntax in expression: {expression}")
            
            context = {**self.context}
            if local_vars:
                context.update(local_vars)
            
            return eval(compile(tree, "<eggscript>", "eval"), {"__builtins__": {}}, context)
        except GRUMError:
            raise
        except Exception as e:
            raise GRUMError(f"Evaluation error: {e} in '{expression}'")

class EggscriptFunction:
    """Represents a user-defined function"""
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class EggscriptSandbox:
    """Enhanced Eggscript execution sandbox"""
    
    def __init__(self):
        self.evaluator = SafeEvaluator()
        self.vars = {'$user_name': 'Canvas User'}
        self.functions = {}
        self.output = ""
        self.script_body = []
        self.line_number = 0
        
        # Helper functions for templates
        self.helpers = {
            'html_escape': lambda s: str(s).replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;'),
            'js_escape': lambda s: str(s).replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'"),
            'url_encode': lambda s: __import__('urllib.parse').quote(str(s)),
            'upper': lambda s: str(s).upper(),
            'lower': lambda s: str(s).lower(),
            'title': lambda s: str(s).title(),
            'strip': lambda s: str(s).strip(),
            'join': lambda items, sep='': sep.join(str(i) for i in items),
        }
    
    def replace_vars(self, text):
        """Replace $variables in text with their values"""
        def replacer(match):
            var_name = match.group(0)
            value = self.vars.get(var_name, var_name)
            return str(value)
        return re.sub(r'\$[a-zA-Z_][a-zA-Z0-9_]*', replacer, text)
    
    def evaluate(self, expression):
        """Evaluate expression with variable substitution"""
        # First replace $vars in expression
        processed = self.replace_vars(expression)
        # Create local context with $ stripped
        local_vars = {k.lstrip('$'): v for k, v in self.vars.items()}
        local_vars.update(self.helpers)
        return self.evaluator.evaluate(processed, local_vars)
    
    def execute_block(self, lines, start_idx=0):
        """Execute a block of lines with control flow support"""
        i = start_idx
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line.startswith('#'):
                i += 1
                continue
            
            try:
                # ~@Show command - output content
                if line.startswith('~@Show '):
                    content = line[7:].strip()
                    result = str(self.evaluate(content))
                    self.output += result + "\n"
                    i += 1
                    continue
                
                # ~@Define Var - variable assignment
                if line.startswith('~@Define Var '):
                    m = re.match(r'~@Define Var\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*)', line, re.IGNORECASE)
                    if m:
                        var_name = "$" + m.group(1)
                        expression = m.group(2).strip()
                        self.vars[var_name] = self.evaluate(expression)
                    i += 1
                    continue
                
                # ~@If conditional
                if line.startswith('~@If '):
                    condition = line[5:].strip()
                    condition_result = self.evaluate(condition)
                    
                    # Find the block to execute
                    if_block, elif_blocks, else_block, endif_idx = self.parse_if_block(lines, i)
                    
                    if condition_result:
                        self.execute_block(if_block)
                    else:
                        executed = False
                        for elif_cond, elif_block in elif_blocks:
                            if self.evaluate(elif_cond):
                                self.execute_block(elif_block)
                                executed = True
                                break
                        if not executed and else_block:
                            self.execute_block(else_block)
                    
                    i = endif_idx + 1
                    continue
                
                # ~@For loop
                if line.startswith('~@For '):
                    m = re.match(r'~@For\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+(.*)', line, re.IGNORECASE)
                    if m:
                        var_name = "$" + m.group(1)
                        iterable_expr = m.group(2).strip()
                        iterable = self.evaluate(iterable_expr)
                        
                        # Find loop body
                        loop_body, endfor_idx = self.parse_for_block(lines, i)
                        
                        # Execute loop
                        for item in iterable:
                            self.vars[var_name] = item
                            self.execute_block(loop_body)
                        
                        i = endfor_idx + 1
                        continue
                
                # ~@While loop
                if line.startswith('~@While '):
                    condition = line[8:].strip()
                    
                    # Find loop body
                    loop_body, endwhile_idx = self.parse_while_block(lines, i)
                    
                    # Execute loop
                    max_iterations = 10000  # Safety limit
                    iterations = 0
                    while self.evaluate(condition) and iterations < max_iterations:
                        self.execute_block(loop_body)
                        iterations += 1
                    
                    if iterations >= max_iterations:
                        raise GRUMError(f"While loop exceeded maximum iterations (line {i+1})")
                    
                    i = endwhile_idx + 1
                    continue
                
                # ~@Function definition
                if line.startswith('~@Function '):
                    func_def, endfunc_idx = self.parse_function(lines, i)
                    self.functions[func_def.name] = func_def
                    i = endfunc_idx + 1
                    continue
                
                # Raw content (HTML/CSS/JS) with variable substitution
                processed_line = self.replace_vars(line)
                self.output += processed_line + "\n"
                
            except GRUMError as e:
                self.output += f"<!-- ERROR on line {i+1}: {e} -->\n"
            
            i += 1
    
    def parse_if_block(self, lines, start_idx):
        """Parse if/elif/else/endif block"""
        if_block = []
        elif_blocks = []
        else_block = []
        
        current_block = if_block
        current_elif_cond = None
        
        i = start_idx + 1
        depth = 1
        
        while i < len(lines) and depth > 0:
            line = lines[i].strip()
            
            if line.startswith('~@If '):
                depth += 1
            elif line.startswith('~@EndIf'):
                depth -= 1
                if depth == 0:
                    break
            elif depth == 1:
                if line.startswith('~@Elif '):
                    if current_elif_cond:
                        elif_blocks.append((current_elif_cond, current_block))
                    current_elif_cond = line[7:].strip()
                    current_block = []
                    i += 1
                    continue
                elif line.startswith('~@Else'):
                    if current_elif_cond:
                        elif_blocks.append((current_elif_cond, current_block))
                        current_elif_cond = None
                    current_block = else_block
                    i += 1
                    continue
            
            current_block.append(lines[i])
            i += 1
        
        if current_elif_cond:
            elif_blocks.append((current_elif_cond, current_block))
        
        return if_block, elif_blocks, else_block, i
    
    def parse_for_block(self, lines, start_idx):
        """Parse for/endfor block"""
        loop_body = []
        i = start_idx + 1
        depth = 1
        
        while i < len(lines) and depth > 0:
            line = lines[i].strip()
            if line.startswith('~@For '):
                depth += 1
            elif line.startswith('~@EndFor'):
                depth -= 1
                if depth == 0:
                    break
            loop_body.append(lines[i])
            i += 1
        
        return loop_body, i
    
    def parse_while_block(self, lines, start_idx):
        """Parse while/endwhile block"""
        loop_body = []
        i = start_idx + 1
        depth = 1
        
        while i < len(lines) and depth > 0:
            line = lines[i].strip()
            if line.startswith('~@While '):
                depth += 1
            elif line.startswith('~@EndWhile'):
                depth -= 1
                if depth == 0:
                    break
            loop_body.append(lines[i])
            i += 1
        
        return loop_body, i
    
    def parse_function(self, lines, start_idx):
        """Parse function definition"""
        line = lines[start_idx].strip()
        m = re.match(r'~@Function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)', line, re.IGNORECASE)
        if not m:
            raise GRUMError(f"Invalid function syntax: {line}")
        
        name = m.group(1)
        params = [p.strip() for p in m.group(2).split(',') if p.strip()]
        
        func_body = []
        i = start_idx + 1
        depth = 1
        
        while i < len(lines) and depth > 0:
            line = lines[i].strip()
            if line.startswith('~@Function '):
                depth += 1
            elif line.startswith('~@EndFunction'):
                depth -= 1
                if depth == 0:
                    break
            func_body.append(lines[i])
            i += 1
        
        return EggscriptFunction(name, params, func_body), i
    
    def parse(self, text):
        """Parse the script into configuration and executable body"""
        lines = text.splitlines()
        in_body = False
        
        for i, line in enumerate(lines):
            l = line.strip()
            
            # Executable Body Start/End
            if l.startswith('~@Start'):
                in_body = True
                continue
            elif l.startswith('~@End'):
                in_body = False
                continue
            
            if in_body:
                self.script_body.append(line)
            else:
                # Pre-body content (config or raw HTML)
                if l and not l.startswith(('~@Define', '~@Add', '!End', 'Begin', 'End')):
                    self.script_body.append(line)
    
    def run(self, text, input_data=None):
        """Execute the eggscript"""
        self.output = ""
        self.vars = {'$user_name': 'Canvas User'}
        self.functions = {}
        self.script_body = []
        
        self.parse(text)
        
        if self.script_body:
            self.execute_block(self.script_body)
        
        if not self.output:
            return "<!-- EGGSCRIPT: No output generated -->"
        
        return self.output

def run_egg_file(path):
    """Run an .egg or .eggless file"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"The file '{path}' does not exist.")
    
    content = p.read_text(encoding='utf-8')
    
    if p.suffix == '.egg':
        content = deobfuscate(content)
    
    sandbox = EggscriptSandbox()
    return sandbox.run(content)

if __name__ == '__main__':
    # Test the eggscript engine
    test_script = """
    ~@Define Var MyTitle = "Eggscript Test Page"
    
    ~@Start
    ~@Define Var Result = 5 + 3
    
    ~@Show "<h1>" + str($MyTitle) + "</h1>"
    ~@Show "<p>Result: " + str($Result) + "</p>"
    
    ~@If $Result > 5
        ~@Show "<p>Result is greater than 5</p>"
    ~@EndIf
    
    ~@For i in range(3)
        ~@Show "<p>Loop iteration: " + str($i) + "</p>"
    ~@EndFor
    ~@End
    """
    try:
        sandbox = EggscriptSandbox()
        print(sandbox.run(test_script))
    except GRUMError as e:
        print(f"Test Failed: {e}")
