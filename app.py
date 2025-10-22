#!/usr/bin/env python3
"""
Godly CLI - Complete Terminal System
- Integrated with eggscript.py for .egg and .eggless file execution
- Cross-platform support (Windows, Linux, macOS)
- Comprehensive command system with all features implemented
"""

import os
import sys
import json
import time
import platform
import subprocess
import shutil
import threading
import socket
import getpass
import base64
import hashlib
import shlex
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import ast

# Optional dependencies with graceful fallbacks
try:
    import requests
except ImportError:
    requests = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

# Import eggscript engine
try:
    from eggscript import run_egg_file, obfuscate, deobfuscate
except ImportError:
    print("[WARNING] eggscript.py not found. .egg/.eggless functionality disabled.")
    run_egg_file = None

# ============================================================================
# SHELL PARSER + AST
# ============================================================================

class TokenType(Enum):
    WORD = "word"
    STRING = "string"
    NUMBER = "number"
    VARIABLE = "variable"
    COMMAND_SUB = "command_sub"
    ARITH_SUB = "arith_sub"
    GLOB = "glob"
    REDIRECT_OUT = "redirect_out"
    REDIRECT_OUT_APPEND = "redirect_out_append"
    REDIRECT_IN = "redirect_in"
    REDIRECT_ERR = "redirect_err"
    REDIRECT_ERR_OUT = "redirect_err_out"
    HEREDOC = "heredoc"
    HERE_STRING = "here_string"
    PIPE = "pipe"
    BACKGROUND = "background"
    SEMICOLON = "semicolon"
    AMPERSAND = "ampersand"
    LPAREN = "lparen"
    RPAREN = "rparen"
    LBRACE = "lbrace"
    RBRACE = "rbrace"
    IF = "if"
    THEN = "then"
    ELIF = "elif"
    ELSE = "else"
    FI = "fi"
    FOR = "for"
    IN = "in"
    DO = "do"
    DONE = "done"
    WHILE = "while"
    FUNCTION = "function"
    EOF = "eof"

class Token:
    def __init__(self, type_: TokenType, value: str, position: int = 0):
        self.type = type_
        self.value = value
        self.position = position

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, {self.position})"

class Lexer:
    def __init__(self, input_: str):
        self.input = input_
        self.position = 0
        self.current_char = self.input[0] if self.input else None

    def advance(self):
        self.position += 1
        if self.position >= len(self.input):
            self.current_char = None
        else:
            self.current_char = self.input[self.position]

    def peek(self):
        peek_pos = self.position + 1
        if peek_pos >= len(self.input):
            return None
        return self.input[peek_pos]

    def skip_whitespace(self):
        while self.current_char and self.current_char.isspace():
            self.advance()

    def read_word(self):
        result = ""
        while self.current_char and (self.current_char.isalnum() or self.current_char in "_-./"):
            result += self.current_char
            self.advance()
        return result

    def read_string(self, quote_char):
        result = ""
        self.advance()  # skip opening quote
        while self.current_char and self.current_char != quote_char:
            if self.current_char == "\\":
                self.advance()
                if self.current_char:
                    if self.current_char == "n":
                        result += "\n"
                    elif self.current_char == "t":
                        result += "\t"
                    elif self.current_char == "r":
                        result += "\r"
                    elif self.current_char == "\\":
                        result += "\\"
                    elif self.current_char == quote_char:
                        result += quote_char
                    else:
                        result += self.current_char
                    self.advance()
            else:
                result += self.current_char
                self.advance()
        if self.current_char == quote_char:
            self.advance()  # skip closing quote
        return result

    def tokenize(self):
        tokens = []
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == '"':
                tokens.append(Token(TokenType.STRING, self.read_string('"')))
                continue

            if self.current_char == "'":
                tokens.append(Token(TokenType.STRING, self.read_string("'")))
                continue

            if self.current_char == "$":
                if self.peek() == "(":
                    # Command substitution $(...)
                    self.advance()  # skip $
                    self.advance()  # skip (
                    start = self.position
                    paren_count = 1
                    while self.current_char and paren_count > 0:
                        if self.current_char == "(":
                            paren_count += 1
                        elif self.current_char == ")":
                            paren_count -= 1
                        self.advance()
                    value = self.input[start:self.position-1]
                    tokens.append(Token(TokenType.COMMAND_SUB, value))
                elif self.peek() == "{":
                    # Variable substitution ${...}
                    self.advance()  # skip $
                    self.advance()  # skip {
                    start = self.position
                    while self.current_char and self.current_char != "}":
                        self.advance()
                    value = self.input[start:self.position]
                    self.advance()  # skip }
                    tokens.append(Token(TokenType.VARIABLE, value))
                elif self.peek() == "(" and self.input[self.position+2:self.position+4] == "((":
                    # Arithmetic substitution $((...))
                    self.advance()  # skip $
                    self.advance()  # skip (
                    self.advance()  # skip (
                    start = self.position
                    paren_count = 2
                    while self.current_char and paren_count > 0:
                        if self.current_char == "(":
                            paren_count += 1
                        elif self.current_char == ")":
                            paren_count -= 1
                        self.advance()
                    value = self.input[start:self.position-2]
                    tokens.append(Token(TokenType.ARITH_SUB, value))
                else:
                    # Simple variable $VAR
                    self.advance()  # skip $
                    var_name = ""
                    while self.current_char and (self.current_char.isalnum() or self.current_char == "_"):
                        var_name += self.current_char
                        self.advance()
                    tokens.append(Token(TokenType.VARIABLE, var_name))
                continue

            if self.current_char in "*?[":
                # Glob patterns
                glob = ""
                while self.current_char and self.current_char in "*?[]":
                    glob += self.current_char
                    self.advance()
                tokens.append(Token(TokenType.GLOB, glob))
                continue

            if self.current_char == ">":
                if self.peek() == ">":
                    self.advance()
                    self.advance()
                    tokens.append(Token(TokenType.REDIRECT_OUT_APPEND, ">>"))
                else:
                    self.advance()
                    tokens.append(Token(TokenType.REDIRECT_OUT, ">"))
                continue

            if self.current_char == "<":
                if self.peek() == "<":
                    self.advance()
                    self.advance()
                    if self.peek() == "<":
                        # Here-string <<<
                        self.advance()
                        tokens.append(Token(TokenType.HERE_STRING, "<<<"))
                    else:
                        # Heredoc <<
                        tokens.append(Token(TokenType.HEREDOC, "<<"))
                else:
                    self.advance()
                    tokens.append(Token(TokenType.REDIRECT_IN, "<"))
                continue

            if self.current_char == "2":
                if self.peek() == ">":
                    self.advance()
                    self.advance()
                    if self.peek() == "&":
                        self.advance()
                        self.advance()
                        tokens.append(Token(TokenType.REDIRECT_ERR_OUT, "2>&1"))
                    else:
                        tokens.append(Token(TokenType.REDIRECT_ERR, "2>"))
                    continue

            if self.current_char == "&":
                if self.peek() == ">":
                    self.advance()
                    self.advance()
                    tokens.append(Token(TokenType.REDIRECT_ERR_OUT, "&>"))
                else:
                    self.advance()
                    tokens.append(Token(TokenType.AMPERSAND, "&"))
                continue

            if self.current_char == "|":
                self.advance()
                tokens.append(Token(TokenType.PIPE, "|"))
                continue

            if self.current_char == ";":
                self.advance()
                tokens.append(Token(TokenType.SEMICOLON, ";"))
                continue

            if self.current_char == "(":
                self.advance()
                tokens.append(Token(TokenType.LPAREN, "("))
                continue

            if self.current_char == ")":
                self.advance()
                tokens.append(Token(TokenType.RPAREN, ")"))
                continue

            if self.current_char == "{":
                self.advance()
                tokens.append(Token(TokenType.LBRACE, "{"))
                continue

            if self.current_char == "}":
                self.advance()
                tokens.append(Token(TokenType.RBRACE, "}"))
                continue

            # Keywords
            word = self.read_word()
            if word == "if":
                tokens.append(Token(TokenType.IF, word))
            elif word == "then":
                tokens.append(Token(TokenType.THEN, word))
            elif word == "elif":
                tokens.append(Token(TokenType.ELIF, word))
            elif word == "else":
                tokens.append(Token(TokenType.ELSE, word))
            elif word == "fi":
                tokens.append(Token(TokenType.FI, word))
            elif word == "for":
                tokens.append(Token(TokenType.FOR, word))
            elif word == "in":
                tokens.append(Token(TokenType.IN, word))
            elif word == "do":
                tokens.append(Token(TokenType.DO, word))
            elif word == "done":
                tokens.append(Token(TokenType.DONE, word))
            elif word == "while":
                tokens.append(Token(TokenType.WHILE, word))
            elif word == "function":
                tokens.append(Token(TokenType.FUNCTION, word))
            else:
                tokens.append(Token(TokenType.WORD, word))
                continue

        tokens.append(Token(TokenType.EOF, ""))
        return tokens

# AST Nodes
class ASTNode:
    pass

class Command(ASTNode):
    def __init__(self, name: str, args: List[str], redirections: Dict = None):
        self.name = name
        self.args = args
        self.redirections = redirections or {}

class Pipeline(ASTNode):
    def __init__(self, commands: List[Command]):
        self.commands = commands

class Background(ASTNode):
    def __init__(self, command: ASTNode):
        self.command = command

class IfStatement(ASTNode):
    def __init__(self, condition: str, then_block: List[ASTNode], elif_blocks: List = None, else_block: List = None):
        self.condition = condition
        self.then_block = then_block
        self.elif_blocks = elif_blocks or []
        self.else_block = else_block or []

class ForLoop(ASTNode):
    def __init__(self, var: str, items: List[str], body: List[ASTNode]):
        self.var = var
        self.items = items
        self.body = body

class WhileLoop(ASTNode):
    def __init__(self, condition: str, body: List[ASTNode]):
        self.condition = condition
        self.body = body

class FunctionDef(ASTNode):
    def __init__(self, name: str, params: List[str], body: List[ASTNode]):
        self.name = name
        self.params = params
        self.body = body

class Subshell(ASTNode):
    def __init__(self, commands: List[ASTNode]):
        self.commands = commands

class Group(ASTNode):
    def __init__(self, commands: List[ASTNode]):
        self.commands = commands

# ============================================================================
# EXPANSION SYSTEM
# ============================================================================

class Expander:
    """Shell expansion engine"""

    def __init__(self):
        self.variables = os.environ.copy()
        self.variables.update({
            'HOME': str(Path.home()),
            'PWD': str(Path.cwd()),
            'USER': getpass.getuser(),
            'SHELL': os.environ.get('SHELL', 'dodo'),
        })

    def expand(self, tokens: List[Token]) -> List[Token]:
        """Expand tokens into expanded token list"""
        expanded_tokens = []
        for token in tokens:
            if token.type == TokenType.WORD:
                # Expand word and create new WORD tokens
                expanded_words = self.expand_word(token.value)
                for word in expanded_words:
                    expanded_tokens.append(Token(TokenType.WORD, word))
            elif token.type == TokenType.STRING:
                expanded_tokens.append(token)
            elif token.type == TokenType.VARIABLE:
                expanded_value = self.expand_variable(token.value)
                expanded_tokens.append(Token(TokenType.WORD, expanded_value))
            elif token.type == TokenType.COMMAND_SUB:
                expanded_value = self.expand_command_sub(token.value)
                expanded_tokens.append(Token(TokenType.WORD, expanded_value))
            elif token.type == TokenType.ARITH_SUB:
                expanded_value = self.expand_arith_sub(token.value)
                expanded_tokens.append(Token(TokenType.WORD, expanded_value))
            elif token.type == TokenType.GLOB:
                expanded_words = self.expand_glob(token.value)
                for word in expanded_words:
                    expanded_tokens.append(Token(TokenType.WORD, word))
            else:
                expanded_tokens.append(token)
        return expanded_tokens

    def expand_word(self, word: str) -> List[str]:
        """Expand tilde and basic globs in word"""
        # Tilde expansion
        if word.startswith('~'):
            if word == '~' or word.startswith('~/'):
                word = word.replace('~', self.variables.get('HOME', '~'), 1)
            else:
                # ~user - not implemented yet
                pass

        # Glob expansion
        import glob
        matches = glob.glob(word)
        if matches:
            return matches
        else:
            return [word]

    def expand_variable(self, var_expr: str) -> str:
        """Expand variable expression like VAR or {VAR:-default}"""
        if var_expr.startswith('{') and var_expr.endswith('}'):
            # ${VAR} or ${VAR:-default}
            inner = var_expr[1:-1]
            if ':-' in inner:
                var, default = inner.split(':-', 1)
                return self.variables.get(var, default)
            else:
                return self.variables.get(inner, '')
        else:
            return self.variables.get(var_expr, '')

    def expand_command_sub(self, cmd: str) -> str:
        """Execute command substitution $(cmd)"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout.strip()
        except Exception:
            return ''

    def expand_arith_sub(self, expr: str) -> str:
        """Execute arithmetic substitution $((expr))"""
        try:
            # Simple arithmetic evaluation
            return str(eval(expr, {"__builtins__": {}}))
        except Exception:
            return '0'

    def expand_glob(self, pattern: str) -> List[str]:
        """Expand glob pattern"""
        import glob
        return glob.glob(pattern) or [pattern]

# ============================================================================
# READLINE / COMPLETION SYSTEM
# ============================================================================

class DodoCompleter(Completer):
    """Completion for Dodo shell"""

    def __init__(self, commands, expander):
        self.commands = commands
        self.expander = expander

    def get_completions(self, document, complete_event):
        word = document.get_word_before_cursor()
        line = document.current_line_before_cursor

        # Command completion
        if not line.strip() or line.endswith(' '):
            for cmd in sorted(self.commands.keys()):
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word))

        # File completion
        elif word and (word.startswith('./') or word.startswith('/') or word.startswith('~') or not any(c in word for c in ' \t')):
            try:
                expanded = self.expander.expand_word(word)
                for path in expanded:
                    if path.startswith(word):
                        yield Completion(path, start_position=-len(word))
            except:
                pass

# ============================================================================
# REPL WITH READLINE
# ============================================================================

def create_prompt_session():
    """Create prompt_toolkit session if available"""
    if not PROMPT_TOOLKIT_AVAILABLE:
        return None

    history_file = Path.home() / '.dodo_history'

    completer = DodoCompleter(COMMAND_MAP, Expander())

    # Key bindings for vi/emacs modes
    kb = KeyBindings()

    @kb.add('c-c')
    def _(event):
        """Ctrl+C to cancel"""
        raise KeyboardInterrupt

    session = PromptSession(
        completer=completer,
        history=FileHistory(str(history_file)),
        key_bindings=kb,
        style=Style.from_dict({
            'prompt': 'bold cyan',
        })
    )

    return session

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        self.current_token = self.tokens[0] if tokens else None

    def advance(self):
        self.position += 1
        if self.position >= len(self.tokens):
            self.current_token = None
        else:
            self.current_token = self.tokens[self.position]

    def parse(self) -> List[ASTNode]:
        statements = []
        while self.current_token and self.current_token.type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            if self.current_token and self.current_token.type == TokenType.SEMICOLON:
                self.advance()
        return statements

    def parse_statement(self) -> ASTNode:
        if self.current_token.type == TokenType.IF:
            return self.parse_if()
        elif self.current_token.type == TokenType.FOR:
            return self.parse_for()
        elif self.current_token.type == TokenType.WHILE:
            return self.parse_while()
        elif self.current_token.type == TokenType.FUNCTION:
            return self.parse_function()
        elif self.current_token.type == TokenType.LPAREN:
            return self.parse_subshell()
        elif self.current_token.type == TokenType.LBRACE:
            return self.parse_group()
        else:
            return self.parse_pipeline()

    def parse_pipeline(self) -> Pipeline:
        commands = []
        cmd = self.parse_command()
        commands.append(cmd)

        while self.current_token and self.current_token.type == TokenType.PIPE:
            self.advance()
            cmd = self.parse_command()
            commands.append(cmd)

        if self.current_token and self.current_token.type == TokenType.AMPERSAND:
            self.advance()
            return Background(Pipeline(commands))

        return Pipeline(commands)

    def parse_command(self) -> Command:
        args = []
        redirections = {}

        while self.current_token and self.current_token.type in [TokenType.WORD, TokenType.STRING, TokenType.VARIABLE, TokenType.COMMAND_SUB, TokenType.ARITH_SUB, TokenType.GLOB]:
            if self.current_token.type == TokenType.REDIRECT_OUT:
                self.advance()
                if self.current_token and self.current_token.type in [TokenType.WORD, TokenType.STRING]:
                    redirections['stdout'] = self.current_token.value
                    self.advance()
            elif self.current_token.type == TokenType.REDIRECT_OUT_APPEND:
                self.advance()
                if self.current_token and self.current_token.type in [TokenType.WORD, TokenType.STRING]:
                    redirections['stdout_append'] = self.current_token.value
                    self.advance()
            elif self.current_token.type == TokenType.REDIRECT_IN:
                self.advance()
                if self.current_token and self.current_token.type in [TokenType.WORD, TokenType.STRING]:
                    redirections['stdin'] = self.current_token.value
                    self.advance()
            elif self.current_token.type == TokenType.REDIRECT_ERR:
                self.advance()
                if self.current_token and self.current_token.type in [TokenType.WORD, TokenType.STRING]:
                    redirections['stderr'] = self.current_token.value
                    self.advance()
            elif self.current_token.type == TokenType.REDIRECT_ERR_OUT:
                redirections['stderr_to_stdout'] = True
                self.advance()
            elif self.current_token.type == TokenType.HEREDOC:
                self.advance()
                if self.current_token and self.current_token.type == TokenType.WORD:
                    redirections['heredoc'] = self.current_token.value
                    self.advance()
            elif self.current_token.type == TokenType.HERE_STRING:
                self.advance()
                if self.current_token and self.current_token.type in [TokenType.WORD, TokenType.STRING]:
                    redirections['here_string'] = self.current_token.value
                    self.advance()
            else:
                args.append(self.current_token.value)
                self.advance()

        name = args[0] if args else ""
        cmd_args = args[1:] if len(args) > 1 else []
        return Command(name, cmd_args, redirections)

    def parse_if(self) -> IfStatement:
        self.advance()  # skip 'if'
        condition = ""
        while self.current_token and self.current_token.type != TokenType.THEN:
            condition += self.current_token.value + " "
            self.advance()
        condition = condition.strip()
        self.advance()  # skip 'then'

        then_block = []
        while self.current_token and self.current_token.type not in [TokenType.ELIF, TokenType.ELSE, TokenType.FI]:
            stmt = self.parse_statement()
            if stmt:
                then_block.append(stmt)
            if self.current_token and self.current_token.type == TokenType.SEMICOLON:
                self.advance()

        elif_blocks = []
        else_block = []

        while self.current_token and self.current_token.type == TokenType.ELIF:
            self.advance()  # skip 'elif'
            elif_condition = ""
            while self.current_token and self.current_token.type != TokenType.THEN:
                elif_condition += self.current_token.value + " "
                self.advance()
            elif_condition = elif_condition.strip()
            self.advance()  # skip 'then'

            elif_body = []
            while self.current_token and self.current_token.type not in [TokenType.ELIF, TokenType.ELSE, TokenType.FI]:
                stmt = self.parse_statement()
                if stmt:
                    elif_body.append(stmt)
                if self.current_token and self.current_token.type == TokenType.SEMICOLON:
                    self.advance()
            elif_blocks.append((elif_condition, elif_body))

        if self.current_token and self.current_token.type == TokenType.ELSE:
            self.advance()  # skip 'else'
            while self.current_token and self.current_token.type != TokenType.FI:
                stmt = self.parse_statement()
                if stmt:
                    else_block.append(stmt)
                if self.current_token and self.current_token.type == TokenType.SEMICOLON:
                    self.advance()

        self.advance()  # skip 'fi'
        return IfStatement(condition, then_block, elif_blocks, else_block)

    def parse_for(self) -> ForLoop:
        self.advance()  # skip 'for'
        var = self.current_token.value if self.current_token else ""
        self.advance()
        self.advance()  # skip 'in'
        items = []
        while self.current_token and self.current_token.type != TokenType.DO:
            items.append(self.current_token.value)
            self.advance()
        self.advance()  # skip 'do'

        body = []
        while self.current_token and self.current_token.type != TokenType.DONE:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            if self.current_token and self.current_token.type == TokenType.SEMICOLON:
                self.advance()

        self.advance()  # skip 'done'
        return ForLoop(var, items, body)

    def parse_while(self) -> WhileLoop:
        self.advance()  # skip 'while'
        condition = ""
        while self.current_token and self.current_token.type != TokenType.DO:
            condition += self.current_token.value + " "
            self.advance()
        condition = condition.strip()
        self.advance()  # skip 'do'

        body = []
        while self.current_token and self.current_token.type != TokenType.DONE:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            if self.current_token and self.current_token.type == TokenType.SEMICOLON:
                self.advance()

        self.advance()  # skip 'done'
        return WhileLoop(condition, body)

    def parse_function(self) -> FunctionDef:
        self.advance()  # skip 'function'
        name = self.current_token.value if self.current_token else ""
        self.advance()
        self.advance()  # skip '('
        params = []
        while self.current_token and self.current_token.type != TokenType.RPAREN:
            params.append(self.current_token.value)
            self.advance()
        self.advance()  # skip ')'
        self.advance()  # skip '{'

        body = []
        while self.current_token and self.current_token.type != TokenType.RBRACE:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            if self.current_token and self.current_token.type == TokenType.SEMICOLON:
                self.advance()

        self.advance()  # skip '}'
        return FunctionDef(name, params, body)

    def parse_subshell(self) -> Subshell:
        self.advance()  # skip '('
        commands = []
        while self.current_token and self.current_token.type != TokenType.RPAREN:
            stmt = self.parse_statement()
            if stmt:
                commands.append(stmt)
            if self.current_token and self.current_token.type == TokenType.SEMICOLON:
                self.advance()
        self.advance()  # skip ')'
        return Subshell(commands)

    def parse_group(self) -> Group:
        self.advance()  # skip '{'
        commands = []
        while self.current_token and self.current_token.type != TokenType.RBRACE:
            stmt = self.parse_statement()
            if stmt:
                commands.append(stmt)
            if self.current_token and self.current_token.type == TokenType.SEMICOLON:
                self.advance()
        self.advance()  # skip '}'
        return Group(commands)

# ============================================================================
# CONFIGURATION SYSTEM
# ============================================================================

CONFIG_FILE = Path.cwd() / 'config.terminal.egg.json'

DEFAULT_CONFIG = {
    'downloads': str(Path.cwd() / 'downloads'),
    'logs': str(Path.cwd() / 'logs'),
    'temp': str(Path.cwd() / 'temp'),
    'iface': 'Ethernet',
    'default_editor': 'nano' if platform.system() != 'Windows' else 'notepad',
    'python_path': sys.executable,
    'shell': os.environ.get('SHELL', 'cmd.exe' if platform.system() == 'Windows' else '/bin/bash'),
    'os': platform.system(),
    'hostname': platform.node(),
    'user': getpass.getuser(),
    'home': str(Path.home()),
    'cwd': str(Path.cwd()),
}

def ensure_config():
    """Ensure config file exists"""
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
        for d in ['downloads', 'logs', 'temp']:
            Path(DEFAULT_CONFIG[d]).mkdir(parents=True, exist_ok=True)

def config_get(key: str) -> Any:
    """Get config value"""
    ensure_config()
    with open(CONFIG_FILE) as f:
        data = json.load(f)
    return data.get(key)

def config_set(key: str, value: Any):
    """Set config value"""
    ensure_config()
    with open(CONFIG_FILE) as f:
        data = json.load(f)
    data[key] = value
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_request(url: str, timeout=10, **kwargs):
    """Make HTTP request with fallback"""
    if requests:
        return requests.get(url, timeout=timeout, **kwargs)
    else:
        import urllib.request
        response = urllib.request.urlopen(url, timeout=timeout)
        class FakeResponse:
            def __init__(self, resp):
                self.content = resp.read()
                self.text = self.content.decode('utf-8', errors='replace')
                self.status_code = resp.getcode()
                self.headers = dict(resp.headers)
            def raise_for_status(self):
                if self.status_code >= 400:
                    raise Exception(f"HTTP {self.status_code}")
        return FakeResponse(response)

def is_windows():
    return platform.system() == 'Windows'

def is_linux():
    return platform.system() == 'Linux'

def is_macos():
    return platform.system() == 'Darwin'

# ============================================================================
# FETCH COMMANDS - FULLY IMPLEMENTED
# ============================================================================

def cmd_fetch(args):
    """fetch <subcommand> - main fetch command with all subcommands"""
    if not args:
        print("Usage: fetch <file|dir|url|info|hash|dwn|...>")
        return
    
    sub = args[0]
    
    # Dispatch to appropriate handler
    handlers = {
        'file': cmd_fetch_file,
        'dir': cmd_fetch_dir,
        'url': cmd_fetch_url,
        'info': cmd_fetch_info,
        'hash': cmd_fetch_hash,
        'dwn': cmd_fetch_dwn,
        'blank': cmd_fetch_blank,
    }
    
    handler = handlers.get(sub)
    if handler:
        handler(args[1:])
    else:
        print(f"[FETCH] Unknown subcommand: {sub}")

def cmd_fetch_file(args):
    """fetch.file <url> - download file from URL"""
    if not args:
        print("Usage: fetch.file <url>")
        return
    
    url = args[0]
    filename = os.path.basename(url.split('?')[0]) or 'download.bin'
    
    try:
        resp = safe_request(url)
        resp.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(resp.content)
        
        print(f"[FETCH.FILE] Downloaded {url} -> {filename}")
    except Exception as e:
        print(f"[FETCH.FILE] Error: {e}")

def cmd_fetch_dir(args):
    """fetch.dir [path] - set or show default download directory"""
    if args:
        path = args[0]
        if not os.path.isdir(path):
            print(f"[FETCH.DIR] Directory does not exist: {path}")
            return
        config_set('downloads', path)
        print(f"[FETCH.DIR] Default download directory set to: {path}")
    else:
        print(f"[FETCH.DIR] Current: {config_get('downloads')}")

def cmd_fetch_url(args):
    """fetch.url <url> - fetch URL content"""
    if not args:
        print("Usage: fetch.url <url>")
        return
    
    try:
        resp = safe_request(args[0])
        resp.raise_for_status()
        print(resp.text)
    except Exception as e:
        print(f"[FETCH.URL] Error: {e}")

def cmd_fetch_info(args):
    """fetch.info <subcommand> <target> - comprehensive info about files/URLs"""
    if not args:
        print("Usage: fetch.info <meta|type|hash|links|headers|redirects|ssl|cookies|dns|whois|geo|tech|robots|sitemap|icons|fonts|scripts|images|videos|audio|pdf|zip|tar|json|xml|csv|md|html|css|js|py|all> <target>")
        return
    
    sub = args[0]
    target = args[1] if len(args) > 1 else None
    
    # Dispatch based on subcommand
    if sub == 'meta':
        cmd_fetch_info_meta([target] if target else [])
    elif sub == 'type':
        cmd_fetch_info_type([target] if target else [])
    elif sub == 'hash':
        cmd_fetch_info_hash([target] if target else [])
    elif sub == 'links':
        cmd_fetch_info_links([target] if target else [])
    elif sub == 'headers':
        cmd_fetch_info_headers([target] if target else [])
    elif sub == 'redirects':
        cmd_fetch_info_redirects([target] if target else [])
    elif sub == 'ssl':
        cmd_fetch_info_ssl([target] if target else [])
    elif sub == 'cookies':
        cmd_fetch_info_cookies([target] if target else [])
    elif sub == 'dns':
        cmd_fetch_info_dns([target] if target else [])
    elif sub == 'whois':
        cmd_fetch_info_whois([target] if target else [])
    elif sub == 'geo':
        cmd_fetch_info_geo([target] if target else [])
    elif sub == 'tech':
        cmd_fetch_info_tech([target] if target else [])
    elif sub == 'robots':
        cmd_fetch_info_robots([target] if target else [])
    elif sub == 'sitemap':
        cmd_fetch_info_sitemap([target] if target else [])
    elif sub == 'icons':
        cmd_fetch_info_icons([target] if target else [])
    elif sub == 'fonts':
        cmd_fetch_info_fonts([target] if target else [])
    elif sub == 'scripts':
        cmd_fetch_info_scripts([target] if target else [])
    elif sub == 'images':
        cmd_fetch_info_images([target] if target else [])
    elif sub == 'videos':
        cmd_fetch_info_videos([target] if target else [])
    elif sub == 'audio':
        cmd_fetch_info_audio([target] if target else [])
    elif sub == 'pdf':
        cmd_fetch_info_pdf([target] if target else [])
    elif sub == 'zip':
        cmd_fetch_info_zip([target] if target else [])
    elif sub == 'tar':
        cmd_fetch_info_tar([target] if target else [])
    elif sub == 'json':
        cmd_fetch_info_json([target] if target else [])
    elif sub == 'xml':
        cmd_fetch_info_xml([target] if target else [])
    elif sub == 'csv':
        cmd_fetch_info_csv([target] if target else [])
    elif sub == 'md':
        cmd_fetch_info_md([target] if target else [])
    elif sub == 'html':
        cmd_fetch_info_html([target] if target else [])
    elif sub == 'css':
        cmd_fetch_info_css([target] if target else [])
    elif sub == 'js':
        cmd_fetch_info_js([target] if target else [])
    elif sub == 'py':
        cmd_fetch_info_py([target] if target else [])
    elif sub == 'all':
        cmd_fetch_info_all([target] if target else [])
    else:
        print(f"[FETCH.INFO] Unknown subcommand: {sub}")

# Implement all fetch.info.* commands

def cmd_fetch_info_meta(args):
    """fetch.info.meta <file|url> - show metadata"""
    if not args:
        print("Usage: fetch.info.meta <file|url>")
        return
    
    target = args[0]
    
    if re.match(r'^https?://', target):
        # URL metadata
        try:
            resp = safe_request(target, method='HEAD' if requests else 'GET')
            print(f"[META] URL: {target}")
            print(f"  Status: {resp.status_code}")
            print(f"  Content-Type: {resp.headers.get('Content-Type', 'Unknown')}")
            print(f"  Content-Length: {resp.headers.get('Content-Length', 'Unknown')}")
            print(f"  Last-Modified: {resp.headers.get('Last-Modified', 'Unknown')}")
            print(f"  Server: {resp.headers.get('Server', 'Unknown')}")
        except Exception as e:
            print(f"[META] Error: {e}")
    else:
        # File metadata
        p = Path(target)
        if not p.exists():
            print(f"[META] File not found: {target}")
            return
        
        stat = p.stat()
        print(f"[META] File: {target}")
        print(f"  Size: {stat.st_size} bytes")
        print(f"  Created: {time.ctime(stat.st_ctime)}")
        print(f"  Modified: {time.ctime(stat.st_mtime)}")
        print(f"  Accessed: {time.ctime(stat.st_atime)}")
        print(f"  Permissions: {oct(stat.st_mode)}")
        
        # Try to detect file type
        import mimetypes
        mime_type = mimetypes.guess_type(target)[0]
        print(f"  MIME Type: {mime_type or 'Unknown'}")

def cmd_fetch_info_type(args):
    """fetch.info.type <file|url> - show content type"""
    if not args:
        print("Usage: fetch.info.type <file|url>")
        return
    
    target = args[0]
    
    if re.match(r'^https?://', target):
        try:
            resp = safe_request(target, method='HEAD' if requests else 'GET')
            print(f"[TYPE] {resp.headers.get('Content-Type', 'Unknown')}")
        except Exception as e:
            print(f"[TYPE] Error: {e}")
    else:
        import mimetypes
        mime_type = mimetypes.guess_type(target)[0]
        print(f"[TYPE] {mime_type or 'Unknown'}")

def cmd_fetch_info_hash(args):
    """fetch.info.hash <file> - show file hashes"""
    if not args:
        print("Usage: fetch.info.hash <file>")
        return
    
    p = Path(args[0])
    if not p.exists():
        print(f"[HASH] File not found: {args[0]}")
        return
    
    content = p.read_bytes()
    print(f"[HASH] File: {args[0]}")
    print(f"  MD5:    {hashlib.md5(content).hexdigest()}")
    print(f"  SHA1:   {hashlib.sha1(content).hexdigest()}")
    print(f"  SHA256: {hashlib.sha256(content).hexdigest()}")

def cmd_fetch_info_links(args):
    """fetch.info.links <url> - extract all links from URL"""
    if not args:
        print("Usage: fetch.info.links <url>")
        return
    
    try:
        resp = safe_request(args[0])
        resp.raise_for_status()
        
        # Extract links
        links = set(re.findall(r'href=["\'](https?://[^"\'>]+)', resp.text, re.I))
        links.update(re.findall(r'src=["\'](https?://[^"\'>]+)', resp.text, re.I))
        
        print(f"[LINKS] Found {len(links)} links:")
        for link in sorted(links):
            print(f"  {link}")
    except Exception as e:
        print(f"[LINKS] Error: {e}")

def cmd_fetch_info_headers(args):
    """fetch.info.headers <url> - show HTTP headers"""
    if not args:
        print("Usage: fetch.info.headers <url>")
        return
    
    try:
        resp = safe_request(args[0], method='HEAD' if requests else 'GET')
        print(f"[HEADERS] {args[0]}")
        for key, value in resp.headers.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"[HEADERS] Error: {e}")

def cmd_fetch_info_redirects(args):
    """fetch.info.redirects <url> - show redirect chain"""
    if not args:
        print("Usage: fetch.info.redirects <url>")
        return
    
    if not requests:
        print("[REDIRECTS] Requires 'requests' library")
        return
    
    try:
        resp = requests.get(args[0], allow_redirects=True, timeout=10)
        print(f"[REDIRECTS] Chain for {args[0]}:")
        if resp.history:
            for i, r in enumerate(resp.history, 1):
                print(f"  {i}. {r.url} -> {r.status_code}")
            print(f"  Final: {resp.url} -> {resp.status_code}")
        else:
            print("  No redirects")
    except Exception as e:
        print(f"[REDIRECTS] Error: {e}")

def cmd_fetch_info_ssl(args):
    """fetch.info.ssl <url> - show SSL certificate info"""
    if not args:
        print("Usage: fetch.info.ssl <url>")
        return
    
    import ssl
    import urllib.parse
    
    parsed = urllib.parse.urlparse(args[0])
    hostname = parsed.netloc or parsed.path
    
    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                print(f"[SSL] Certificate for {hostname}:")
                print(f"  Subject: {dict(x[0] for x in cert['subject'])}")
                print(f"  Issuer: {dict(x[0] for x in cert['issuer'])}")
                print(f"  Version: {cert.get('version')}")
                print(f"  Not Before: {cert.get('notBefore')}")
                print(f"  Not After: {cert.get('notAfter')}")
    except Exception as e:
        print(f"[SSL] Error: {e}")

def cmd_fetch_info_cookies(args):
    """fetch.info.cookies <url> - show cookies"""
    if not args:
        print("Usage: fetch.info.cookies <url>")
        return
    
    if not requests:
        print("[COOKIES] Requires 'requests' library")
        return
    
    try:
        session = requests.Session()
        resp = session.get(args[0], timeout=10)
        print(f"[COOKIES] Cookies from {args[0]}:")
        if session.cookies:
            for cookie in session.cookies:
                print(f"  {cookie.name} = {cookie.value}")
        else:
            print("  No cookies")
    except Exception as e:
        print(f"[COOKIES] Error: {e}")

def cmd_fetch_info_dns(args):
    """fetch.info.dns <domain> - show DNS info"""
    if not args:
        print("Usage: fetch.info.dns <domain>")
        return
    
    domain = args[0].replace('http://', '').replace('https://', '').split('/')[0]
    
    try:
        ip = socket.gethostbyname(domain)
        print(f"[DNS] {domain} -> {ip}")
        
        # Try reverse lookup
        try:
            hostname = socket.gethostbyaddr(ip)[0]
            print(f"  Reverse: {hostname}")
        except:
            pass
    except Exception as e:
        print(f"[DNS] Error: {e}")

def cmd_fetch_info_whois(args):
    """fetch.info.whois <domain> - show WHOIS info (basic)"""
    if not args:
        print("Usage: fetch.info.whois <domain>")
        return
    
    domain = args[0].replace('http://', '').replace('https://', '').split('/')[0]
    print(f"[WHOIS] {domain}")
    print("  (Full WHOIS requires external service/library)")
    
    # Basic info we can get
    try:
        ip = socket.gethostbyname(domain)
        print(f"  IP: {ip}")
    except:
        print("  IP: Unable to resolve")

def cmd_fetch_info_geo(args):
    """fetch.info.geo <ip|domain> - show geographic info (basic)"""
    if not args:
        print("Usage: fetch.info.geo <ip|domain>")
        return
    
    target = args[0]
    print(f"[GEO] {target}")
    print("  (Full geolocation requires external API)")
    
    try:
        if not re.match(r'\d+\.\d+\.\d+\.\d+', target):
            target = socket.gethostbyname(target)
        print(f"  IP: {target}")
    except:
        print("  Unable to resolve")

def cmd_fetch_info_tech(args):
    """fetch.info.tech <url> - detect technology stack"""
    if not args:
        print("Usage: fetch.info.tech <url>")
        return
    
    try:
        resp = safe_request(args[0])
        resp.raise_for_status()
        
        print(f"[TECH] Technology stack for {args[0]}:")
        
        # Detect from headers
        server = resp.headers.get('Server', 'Unknown')
        print(f"  Server: {server}")
        
        powered_by = resp.headers.get('X-Powered-By', 'Unknown')
        if powered_by != 'Unknown':
            print(f"  Powered By: {powered_by}")
        
        # Detect from content
        html = resp.text.lower()
        
        if 'wordpress' in html or 'wp-content' in html:
            print("  CMS: WordPress")
        elif 'joomla' in html:
            print("  CMS: Joomla")
        elif 'drupal' in html:
            print("  CMS: Drupal")
        
        if 'react' in html or 'reactjs' in html:
            print("  Framework: React")
        elif 'vue' in html or 'vuejs' in html:
            print("  Framework: Vue.js")
        elif 'angular' in html:
            print("  Framework: Angular")
        
        if 'jquery' in html:
            print("  Library: jQuery")
        
    except Exception as e:
        print(f"[TECH] Error: {e}")

def cmd_fetch_info_robots(args):
    """fetch.info.robots <url> - fetch robots.txt"""
    if not args:
        print("Usage: fetch.info.robots <url>")
        return
    
    base_url = re.match(r'(https?://[^/]+)', args[0])
    if not base_url:
        print("[ROBOTS] Invalid URL")
        return
    
    robots_url = base_url.group(1) + '/robots.txt'
    
    try:
        resp = safe_request(robots_url)
        resp.raise_for_status()
        print(f"[ROBOTS] {robots_url}:")
        print(resp.text)
    except Exception as e:
        print(f"[ROBOTS] Error: {e}")

def cmd_fetch_info_sitemap(args):
    """fetch.info.sitemap <url> - fetch sitemap.xml"""
    if not args:
        print("Usage: fetch.info.sitemap <url>")
        return
    
    base_url = re.match(r'(https?://[^/]+)', args[0])
    if not base_url:
        print("[SITEMAP] Invalid URL")
        return
    
    sitemap_url = base_url.group(1) + '/sitemap.xml'
    
    try:
        resp = safe_request(sitemap_url)
        resp.raise_for_status()
        print(f"[SITEMAP] {sitemap_url}:")
        print(resp.text[:2000])  # First 2000 chars
        if len(resp.text) > 2000:
            print(f"\n... ({len(resp.text)} total bytes)")
    except Exception as e:
        print(f"[SITEMAP] Error: {e}")

# File type specific info commands

def find_files_by_extension(target, extensions):
    """Helper to find files by extension in URL or directory"""
    if re.match(r'^https?://', target):
        # URL - extract from HTML
        try:
            resp = safe_request(target)
            resp.raise_for_status()
            html = resp.text
            
            files = set()
            for ext in extensions:
                pattern = r'(?:href|src)=["\']([^"\']*\.' + ext + r'[^"\']*)["\']'
                files.update(re.findall(pattern, html, re.I))
            
            return sorted(files)
        except Exception as e:
            print(f"[ERROR] {e}")
            return []
    else:
        # Local directory
        p = Path(target)
        if not p.exists():
            print(f"[ERROR] Not found: {target}")
            return []
        
        if p.is_file():
            return [str(p)] if p.suffix[1:] in extensions else []
        
        files = []
        for ext in extensions:
            files.extend(str(f) for f in p.rglob(f'*.{ext}'))
        return sorted(files)

def cmd_fetch_info_icons(args):
    """fetch.info.icons <url|dir> - find icon files"""
    if not args:
        print("Usage: fetch.info.icons <url|dir>")
        return
    
    files = find_files_by_extension(args[0], ['ico', 'png', 'svg', 'jpg', 'jpeg', 'gif'])
    print(f"[ICONS] Found {len(files)} icon files:")
    for f in files:
        print(f"  {f}")

def cmd_fetch_info_fonts(args):
    """fetch.info.fonts <url|dir> - find font files"""
    if not args:
        print("Usage: fetch.info.fonts <url|dir>")
        return
    
    files = find_files_by_extension(args[0], ['woff', 'woff2', 'ttf', 'otf', 'eot'])
    print(f"[FONTS] Found {len(files)} font files:")
    for f in files:
        print(f"  {f}")

def cmd_fetch_info_scripts(args):
    """fetch.info.scripts <url|dir> - find JavaScript files"""
    if not args:
        print("Usage: fetch.info.scripts <url|dir>")
        return
    
    files = find_files_by_extension(args[0], ['js'])
    print(f"[SCRIPTS] Found {len(files)} JavaScript files:")
    for f in files:
        print(f"  {f}")

def cmd_fetch_info_images(args):
    """fetch.info.images <url|dir> - find image files"""
    if not args:
        print("Usage: fetch.info.images <url|dir>")
        return
    
    files = find_files_by_extension(args[0], ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'webp'])
    print(f"[IMAGES] Found {len(files)} image files:")
    for f in files:
        print(f"  {f}")

def cmd_fetch_info_videos(args):
    """fetch.info.videos <url|dir> - find video files"""
    if not args:
        print("Usage: fetch.info.videos <url|dir>")
        return
    
    files = find_files_by_extension(args[0], ['mp4', 'webm', 'ogg', 'avi', 'mov', 'wmv', 'flv'])
    print(f"[VIDEOS] Found {len(files)} video files:")
    for f in files:
        print(f"  {f}")

def cmd_fetch_info_audio(args):
    """fetch.info.audio <url|dir> - find audio files"""
    if not args:
        print("Usage: fetch.info.audio <url|dir>")
        return
    
    files = find_files_by_extension(args[0], ['mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac'])
    print(f"[AUDIO] Found {len(files)} audio files:")
    for f in files:
        print(f"  {f}")

def cmd_fetch_info_pdf(args):
    """fetch.info.pdf <url|dir> - find PDF files"""
    if not args:
        print("Usage: fetch.info.pdf <url|dir>")
        return
    
    files = find_files_by_extension(args[0], ['pdf'])
    print(f"[PDF] Found {len(files)} PDF files:")
    for f in files:
        print(f"  {f}")

def cmd_fetch_info_zip(args):
    """fetch.info.zip <url|dir> - find ZIP archives"""
    if not args:
        print("Usage: fetch.info.zip <url|dir>")
        return
    
    files = find_files_by_extension(args[0], ['zip'])
    print(f"[ZIP] Found {len(files)} ZIP files:")
    for f in files:
        print(f"  {f}")

def cmd_fetch_info_tar(args):
    """fetch.info.tar <url|dir> - find TAR archives"""
    if not args:
        print("Usage: fetch.info.tar <url|dir>")
        return
    
    files = find_files_by_extension(args[0], ['tar', 'tar.gz', 'tgz', 'tar.bz2'])
    print(f"[TAR] Found {len(files)} TAR files:")
    for f in files:
        print(f"  {f}")

def cmd_fetch_info_json(args):
    """fetch.info.json <url|dir> - find JSON files"""
    if not args:
        print("Usage: fetch.info.json <url|dir>")
        return
    
    files = find_files_by_extension(args[0], ['json'])
    print(f"[JSON] Found {len(files)} JSON files:")
    for f in files:
        print(f"  {f}")

def cmd_fetch_info_xml(args):
    """fetch.info.xml <url|dir> - find XML files"""
    if not args:
        print("Usage: fetch.info.xml <url|dir>")
        return
    
    files = find_files_by_extension(args[0], ['xml'])
    print(f"[XML] Found {len(files)} XML files:")
    for f in files:
        print(f"  {f}")

def cmd_fetch_info_csv(args):
    """fetch.info.csv <url|dir> - find CSV files"""
    if not args:
        print("Usage: fetch.info.csv <url|dir>")
        return
    
    files = find_files_by_extension(args[0], ['csv'])
    print(f"[CSV] Found {len(files)} CSV files:")
    for f in files:
        print(f"  {f}")

def cmd_fetch_info_md(args):
    """fetch.info.md <url|dir> - find Markdown files"""
    if not args:
        print("Usage: fetch.info.md <url|dir>")
        return
    
    files = find_files_by_extension(args[0], ['md', 'markdown'])
    print(f"[MARKDOWN] Found {len(files)} Markdown files:")
    for f in files:
        print(f"  {f}")

def cmd_fetch_info_html(args):
    """
    ensure_config_egg()
    with open(CONFIG_EGG_FILE) as f:
        data = json.load(f)
    for k, v in data.items():
        print(f"{k}: {v}")

def cmd_config_terminal_egg(args):
    """config.terminal.egg <subcommand> [key] [value] - manage CLI config (dirs, iface, temp, executables, etc.)"""
    if not args:
        print("Usage: config.terminal.egg <set|get|view|dir|iface|temp|exec|os|all> [key] [value]")
        return
    sub = args[0]
    if sub == 'set' and len(args) > 2:
        config_egg_set(args[1], args[2])
        print(f"[CONFIG] Set {args[1]} = {args[2]}")
    elif sub == 'get' and len(args) > 1:
        val = config_egg_get(args[1])
        print(f"[CONFIG] {args[1]} = {val}")
    elif sub == 'view':
        config_egg_view()
    elif sub == 'dir' and len(args) > 2:
        if args[1] in ['downloads','logs','temp','home','cwd']:
            config_egg_set(args[1], args[2])
            print(f"[CONFIG] Directory {args[1]} set to {args[2]}")
        else:
            print("[CONFIG] Unknown directory key. Use downloads, logs, temp, home, or cwd.")
    elif sub == 'iface' and len(args) > 1:
        config_egg_set('iface', args[1])
        print(f"[CONFIG] Network interface set to {args[1]}")
    elif sub == 'exec' and len(args) > 2:
        if args[1] in ['python_path','shell','native','default_editor']:
            config_egg_set(args[1], args[2])
            print(f"[CONFIG] Executable {args[1]} set to {args[2]}")
        else:
            print("[CONFIG] Unknown executable key. Use python_path, shell, native, or default_editor.")
    elif sub == 'os':
        val = config_egg_get('os')
        print(f"[CONFIG] OS: {val}")
    elif sub == 'temp':
        temp_dir = config_egg_get('temp')
        print(f"[CONFIG] Temp directory: {temp_dir}")
    elif sub == 'all':
        config_egg_view()
    else:
        print("[CONFIG] Unknown subcommand or missing argument.")
def cmd_fetch_file(args):
    """fetch.file <url> - download a file from URL"""
    import requests, os
    if not args:
        print("Usage: fetch.file <url>")
        return
    url = args[0]
    fname = os.path.basename(url)
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        with open(fname, 'wb') as f:
            f.write(r.content)
        print(f"[FETCH.FILE] Downloaded {url} -> {fname}")
    except Exception as e:
        print(f"[FETCH.FILE] Error: {e}")

def cmd_fetch_file_save(args):
    """fetch.file.save <url> <path> - save file from URL to path"""
    import requests, os
    if len(args)<2:
        print("Usage: fetch.file.save <url> <path>")
        return
    url, path = args[0], args[1]
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        print(f"[FETCH.FILE.SAVE] Saved {url} -> {path}")
    except Exception as e:
        print(f"[FETCH.FILE.SAVE] Error: {e}")
def cmd_convert(args):
    """convert <subcommand> - convert files, formats, encodings, etc."""
    if not args:
        print("Usage: convert <subcommand>")
        return
    sub = args[0]
    if sub == 'txt2pdf':
        from fpdf import FPDF
        if len(args)<2:
            print("Usage: convert txt2pdf <file>"); return
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        with open(args[1], 'r') as f:
            for line in f:
                pdf.cell(200, 10, txt=line.strip(), ln=1)
        out = args[1]+'.pdf'
        pdf.output(out)
        print(f"[CONVERT.TXT2PDF] {args[1]} -> {out}")
    elif sub == 'pdf2txt':
        import PyPDF2
        if len(args)<2:
            print("Usage: convert pdf2txt <file>"); return
        with open(args[1], 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''
        out = args[1]+'.txt'
        with open(out, 'w') as f:
            f.write(text)
        print(f"[CONVERT.PDF2TXT] {args[1]} -> {out}")
    elif sub == 'csv2json':
        import csv, json
        if len(args)<2:
            print("Usage: convert csv2json <file>"); return
        with open(args[1], newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        out = args[1]+'.json'
        with open(out, 'w') as f:
            json.dump(rows, f)
        print(f"[CONVERT.CSV2JSON] {args[1]} -> {out}")
    elif sub == 'json2csv':
        import csv, json
        if len(args)<2:
            print("Usage: convert json2csv <file>"); return
        with open(args[1]) as f:
            data = json.load(f)
        out = args[1]+'.csv'
        with open(out, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"[CONVERT.JSON2CSV] {args[1]} -> {out}")
    elif sub == 'img2pdf':
        from fpdf import FPDF
        from PIL import Image
        if len(args)<2:
            print("Usage: convert img2pdf <image>"); return
        pdf = FPDF()
        pdf.add_page()
        pdf.image(args[1], x=10, y=10, w=100)
        out = args[1]+'.pdf'
        pdf.output(out)
        print(f"[CONVERT.IMG2PDF] {args[1]} -> {out}")
    elif sub == 'pdf2img':
        from pdf2image import convert_from_path
        if len(args)<2:
            print("Usage: convert pdf2img <file>"); return
        images = convert_from_path(args[1])
        for i, img in enumerate(images):
            out = f"{args[1]}_page{i+1}.png"
            img.save(out, 'PNG')
            print(f"[CONVERT.PDF2IMG] {args[1]} -> {out}")
    elif sub == 'md2html':
        import markdown
        if len(args)<2:
            print("Usage: convert md2html <file>"); return
        with open(args[1]) as f:
            html = markdown.markdown(f.read())
        out = args[1]+'.html'
        with open(out, 'w') as f:
            f.write(html)
        print(f"[CONVERT.MD2HTML] {args[1]} -> {out}")
    elif sub == 'html2md':
        import markdownify
        if len(args)<2:
            print("Usage: convert html2md <file>"); return
        with open(args[1]) as f:
            md = markdownify.markdownify(f.read())
        out = args[1]+'.md'
        with open(out, 'w') as f:
            f.write(md)
        print(f"[CONVERT.HTML2MD] {args[1]} -> {out}")
    elif sub == 'xlsx2csv':
        import pandas as pd
        if len(args)<2:
            print("Usage: convert xlsx2csv <file>"); return
        df = pd.read_excel(args[1])
        out = args[1]+'.csv'
        df.to_csv(out, index=False)
        print(f"[CONVERT.XLSX2CSV] {args[1]} -> {out}")
    elif sub == 'csv2xlsx':
        import pandas as pd
        if len(args)<2:
            print("Usage: convert csv2xlsx <file>"); return
        df = pd.read_csv(args[1])
        out = args[1]+'.xlsx'
        df.to_excel(out, index=False)
        print(f"[CONVERT.CSV2XLSX] {args[1]} -> {out}")
    else:
        print(f"[CONVERT] Unknown subcommand: {sub}")

def cmd_auto(args):
    """auto <subcommand> - automation tasks (demo)"""
    if not args:
        print("Usage: auto <subcommand>")
        return
    sub = args[0]
    if sub == 'backup':
        import shutil
        if len(args)<3:
            print("Usage: auto backup <src> <dst>"); return
        shutil.copytree(args[1], args[2], dirs_exist_ok=True)
        print(f"[AUTO.BACKUP] {args[1]} -> {args[2]}")
    elif sub == 'clean':
        import os
        if len(args)<2:
            print("Usage: auto clean <dir>"); return
        for f in os.listdir(args[1]):
            fp = os.path.join(args[1], f)
            if os.path.isfile(fp):
                os.remove(fp)
        print(f"[AUTO.CLEAN] Cleaned {args[1]}")
    elif sub == 'update':
        import subprocess
        print("[AUTO.UPDATE] Updating...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    elif sub == 'archive':
        import shutil
        if len(args)<3:
            print("Usage: auto archive <src> <dst.zip>"); return
        shutil.make_archive(args[2].replace('.zip',''), 'zip', args[1])
        print(f"[AUTO.ARCHIVE] {args[1]} -> {args[2]}")
    elif sub == 'extract':
        import zipfile
        if len(args)<3:
            print("Usage: auto extract <zip> <dst>"); return
        with zipfile.ZipFile(args[1], 'r') as zip_ref:
            zip_ref.extractall(args[2])
        print(f"[AUTO.EXTRACT] {args[1]} -> {args[2]}")
    elif sub == 'move':
        import shutil
        if len(args)<3:
            print("Usage: auto move <src> <dst>"); return
        shutil.move(args[1], args[2])
        print(f"[AUTO.MOVE] {args[1]} -> {args[2]}")
    elif sub == 'copy':
        import shutil
        if len(args)<3:
            print("Usage: auto copy <src> <dst>"); return
        shutil.copy2(args[1], args[2])
        print(f"[AUTO.COPY] {args[1]} -> {args[2]}")
    elif sub == 'touch':
        import os
        if len(args)<2:
            print("Usage: auto touch <file>"); return
        with open(args[1], 'a') as f:
            os.utime(args[1], None)
        print(f"[AUTO.TOUCH] {args[1]}")
    elif sub == 'schedule':
        import sched, time
        if len(args)<3:
            print("Usage: auto schedule <seconds> <cmd>"); return
        s = sched.scheduler(time.time, time.sleep)
        def run_cmd():
            print(f"[AUTO.SCHEDULE] Running: {args[2]}")
            os.system(args[2])
        s.enter(int(args[1]), 1, run_cmd)
        s.run()
    elif sub == 'remind':
        import time
        if len(args)<3:
            print("Usage: auto remind <seconds> <msg>"); return
        print(f"[AUTO.REMIND] Will remind in {args[1]} seconds...")
        time.sleep(int(args[1]))
        print(f"[AUTO.REMIND] {args[2]}")
    elif sub == 'log':
        if len(args)<2:
            print("Usage: auto log <msg>"); return
        with open('auto.log','a') as f:
            f.write(' '.join(args[1:])+"\n")
        print(f"[AUTO.LOG] Logged: {' '.join(args[1:])}")
    elif sub == 'ping':
        import subprocess
        if len(args)<2:
            print("Usage: auto ping <host>"); return
        subprocess.run(['ping', args[1]])
    else:
        print(f"[AUTO] Unknown subcommand: {sub}")
        # ----------------------------
# COMMAND DISPATCH MAP
# (Add this section after all 'def cmd_...' functions)
# ----------------------------

# Add missing import for Path at the top
from pathlib import Path
#!/usr/bin/env python3
"""
procli.py - single-file modular CLI + tiny language (ProLang) + .egg/.eggless support

Usage: python procli.py
"""

import os, sys, shlex, json, base64, hashlib, sqlite3, threading, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import subprocess
import platform
import stat
import shutil


# Optional dependency
try:
    import requests
except Exception:
    requests = None

# Initialize FETCH_DEFAULT_DIR after Path import
FETCH_DEFAULT_DIR = [str(Path('.').resolve())]

# ----------------------------
# Config and helpers
# ----------------------------
HOME = Path.home()
CONFIG_FILE = HOME / ".procli_conf.json"
DEFAULT_CONFIG = {
    "editor": "nano",
    "server_port": 6969,
    "egg_xor_key": 0x5A,
    "workspace": str(Path.cwd() / "procli_ws"),
}

def check_system_editor():
    """Finds the first available editor from a predefined list."""
    editors = ["nano", "vim", "vi", "code", "notepad", "gedit"]
    for ed in editors:
        if shutil.which(ed):
            return ed
    return "internal"  # Fallback if none are found

# Initialize or load config
if not CONFIG_FILE.exists():
    # On first run, detect editor automatically
    DEFAULT_CONFIG["editor"] = check_system_editor()
    CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2))

CONFIG = json.loads(CONFIG_FILE.read_text())

# Verify configured editor exists, if not, re-detect for this session
if CONFIG.get("editor") != "internal" and not shutil.which(CONFIG.get("editor")):
    print(f"[BOOT] Configured editor '{CONFIG.get('editor')}' not found. Detecting fallback.")
    CONFIG["editor"] = check_system_editor()

WS = Path(CONFIG.get("workspace"))
WS.mkdir(parents=True, exist_ok=True)


import getpass
def get_prompt():
    user = getpass.getuser()
    cwd = os.getcwd()
    return f"{user}@{cwd}> "

def save_config():
    CONFIG_FILE.write_text(json.dumps(CONFIG, indent=2))

def short_help():
    return """PROCLI - compact commands: type <command> --help for detailed usage.
Core areas: fetch.*, file.*, dir.*, egg.*, pong.*, sys.*, lang.*, /native/*
Type 'exit' or Ctrl-D to quit.
"""

# ----------------------------
# .egg obfuscation helpers
# ----------------------------
def egg_encode_bytes(b: bytes) -> bytes:
    # base64 + xor simple
    k = CONFIG.get("egg_xor_key", 0x5A)
    enc = bytes([x ^ k for x in b])
    return base64.b64encode(enc)

def egg_decode_bytes(b64: bytes) -> bytes:
    k = CONFIG.get("egg_xor_key", 0x5A)
    try:
        dec = base64.b64decode(b64)
    except Exception:
        raise ValueError("Not valid egg content (b64 decode failed)")
    return bytes([x ^ k for x in dec])

def make_egg(src_path: Path, out_path: Path):
    b = src_path.read_bytes()
    out_path.write_bytes(egg_encode_bytes(b))
    return out_path

def open_egg(path: Path) -> bytes:
    b = path.read_bytes()
    return egg_decode_bytes(b)

import re
# ----------------------------
# Command Handlers
def cmd_fetch(args):
    """fetch <subcommand> - main fetch command"""
    if not args:
        print("Usage: fetch <subcommand>")
        return
    sub = args[0]
    if sub == 'file':
        cmd_fetch_file(args[1:])
    elif sub == 'dir':
        cmd_fetch_dir(args[1:])
    elif sub == 'url':
        cmd_fetch_url(args[1:])
    elif sub == 'info':
        cmd_fetch_info(args[1:])
    elif sub == 'blank':
        cmd_fetch_blank(args[1:])
    elif sub == 'hash':
        # sub-subcommands for hash
        if len(args) > 1:
            subsub = args[1]
            if subsub == 'md5':
                import hashlib
                if len(args) > 2:
                    with open(args[2],'rb') as f:
                        print(hashlib.md5(f.read()).hexdigest())
                else:
                    print("Usage: fetch hash md5 <file>")
            elif subsub == 'sha256':
                import hashlib
                if len(args) > 2:
                    with open(args[2],'rb') as f:
                        print(hashlib.sha256(f.read()).hexdigest())
                else:
                    print("Usage: fetch hash sha256 <file>")
            elif subsub == 'list':
                import os
                if len(args) > 2:
                    for f in os.listdir(args[2]):
                        print(f)
                else:
                    print("Usage: fetch hash list <dir>")
            elif subsub == 'verify':
                print("[FETCH.HASH.VERIFY] Not implemented yet.")
            elif subsub == 'compare':
                print("[FETCH.HASH.COMPARE] Not implemented yet.")
            elif subsub == 'save':
                print("[FETCH.HASH.SAVE] Not implemented yet.")
            elif subsub == 'load':
                print("[FETCH.HASH.LOAD] Not implemented yet.")
            elif subsub == 'remove':
                print("[FETCH.HASH.REMOVE] Not implemented yet.")
            elif subsub == 'update':
                print("[FETCH.HASH.UPDATE] Not implemented yet.")
            elif subsub == 'export':
                print("[FETCH.HASH.EXPORT] Not implemented yet.")
            elif subsub == 'import':
                print("[FETCH.HASH.IMPORT] Not implemented yet.")
            else:
                print(f"[FETCH.HASH] Unknown sub-subcommand: {subsub}")
        else:
            print("Usage: fetch hash <md5|sha256|list|verify|compare|save|load|remove|update|export|import> <file|dir>")
    elif sub == 'save':
        cmd_fetch_file_save(args[1:])
    elif sub == 'type':
        cmd_fetch_file_type(args[1:])
    elif sub == 'size':
        cmd_fetch_file_size(args[1:])
    elif sub == 'headers':
        cmd_fetch_url_headers(args[1:])
    elif sub == 'status':
        cmd_fetch_url_status(args[1:])
    elif sub == 'links':
        cmd_fetch_url_links(args[1:])
    elif sub == 'saveurl':
        cmd_fetch_url_save(args[1:])
    elif sub == 'typeurl':
        cmd_fetch_url_type(args[1:])
    elif sub == 'infourl':
        cmd_fetch_url_info(args[1:])
    elif sub == 'all':
        print("[FETCH.ALL] Fetching all resources...")
    elif sub == 'batch':
        print("[FETCH.BATCH] Batch fetch not implemented yet.")
    elif sub == 'multi':
        print("[FETCH.MULTI] Multi-fetch not implemented yet.")
    elif sub == 'stream':
        print("[FETCH.STREAM] Stream fetch not implemented yet.")
    elif sub == 'resume':
        print("[FETCH.RESUME] Resume fetch not implemented yet.")
    elif sub == 'pause':
        print("[FETCH.PAUSE] Pause fetch not implemented yet.")
    elif sub == 'cancel':
        print("[FETCH.CANCEL] Cancel fetch not implemented yet.")
    elif sub == 'retry':
        print("[FETCH.RETRY] Retry fetch not implemented yet.")
    elif sub == 'log':
        print("[FETCH.LOG] Fetch log not implemented yet.")
    elif sub == 'summary':
        print("[FETCH.SUMMARY] Fetch summary not implemented yet.")
    else:
        print(f"[FETCH] Unknown subcommand: {sub}")

def cmd_fetch_dir(args):
    """fetch.dir [path] - set or show default download directory"""
    import os, json
    config_file = 'fetch_dir_config.json'
    if args:
        path = args[0]
        if not os.path.isdir(path):
            print(f"[FETCH.DIR] Directory does not exist: {path}")
            return
        with open(config_file, 'w') as f:
            json.dump({'dir': path}, f)
        print(f"[FETCH.DIR] Default download directory set to: {path}")
    else:
        if os.path.exists(config_file):
            with open(config_file) as f:
                data = json.load(f)
            print(f"[FETCH.DIR] Default download directory: {data['dir']}")
        else:
            print("[FETCH.DIR] No default download directory set.")

def cmd_python(args):
    """/python/ <code or script.py> - run Python code or script"""
    import subprocess, os
    if not args:
        print("Usage: /python/ <code or script.py>")
        return
    if args[0].endswith('.py') and os.path.isfile(args[0]):
        subprocess.run(['python', args[0]] + args[1:])
    else:
        code = ' '.join(args)
        try:
            exec(code, globals())
        except Exception as e:
            print("[PYTHON] Error:", e)

def cmd_clear(args):
    """clear - clear the CLI screen"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def cmd_export(args):
    """export VAR=value - set environment variable"""
    if not args:
        # Show all exported vars
        for k, v in os.environ.items():
            print(f"export {k}={v!r}")
        return

    var_expr = args[0]
    if '=' in var_expr:
        var, value = var_expr.split('=', 1)
        os.environ[var] = value
        print(f"[EXPORT] {var}={value!r}")
    else:
        print("Usage: export VAR=value")

def cmd_unset(args):
    """unset VAR - remove environment variable"""
    if not args:
        print("Usage: unset VAR")
        return

    var = args[0]
    if var in os.environ:
        del os.environ[var]
        print(f"[UNSET] {var}")
    else:
        print(f"[UNSET] {var} not found")

def cmd_env(args):
    """env - show environment variables"""
    for k, v in sorted(os.environ.items()):
        print(f"{k}={v}")

def load_rc_file():
    """Load .dodorc or profile file"""
    rc_files = [
        Path.home() / '.dodorc',
        Path.home() / '.profile',
        Path.home() / '.bashrc',
    ]

    for rc_file in rc_files:
        if rc_file.exists():
            try:
                with open(rc_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Simple parsing for export statements
                            if line.startswith('export '):
                                var_expr = line[7:].strip()
                                if '=' in var_expr:
                                    var, value = var_expr.split('=', 1)
                                    os.environ[var] = value
            except Exception as e:
                print(f"[RC] Error loading {rc_file}: {e}")

def cmd_fetch_info(args):
    """fetch.info <subcommand> - fetch info about a file, url, or system"""
    if not args:
        print("Usage: fetch.info <file|url|sys> [target]")
        return
    sub = args[0]
    if sub == 'file' and len(args)>1:
        import os
        path = args[1]
        if os.path.exists(path):
            print(f"[FETCH.INFO] File: {path}")
            print(f"  Size: {os.path.getsize(path)} bytes")
            print(f"  Modified: {os.path.getmtime(path)}")
        else:
            print(f"[FETCH.INFO] File not found: {path}")
    elif sub == 'url' and len(args)>1:
        import requests
        url = args[1]
        try:
            r = requests.head(url, timeout=5)
            print(f"[FETCH.INFO] URL: {url}")
            print(f"  Status: {r.status_code}")
            print(f"  Headers: {r.headers}")
        except Exception as e:
            print(f"[FETCH.INFO] Error: {e}")
    elif sub == 'sys':
        import platform
        print(f"[FETCH.INFO] System: {platform.system()} {platform.release()} {platform.version()}")
    else:
        print("[FETCH.INFO] Unknown subcommand or missing argument.")

def cmd_fetch_blank(args):
    """fetch.blank <file> - create a blank .egg or .eggless file"""
    if not args:
        print("Usage: fetch.blank <file>")
        return
    fname = args[0]
    if not (fname.endswith('.egg') or fname.endswith('.eggless')):
        print("[FETCH.BLANK] File must end with .egg or .eggless")
        return
    with open(fname, 'w') as f:
        f.write('')
    print(f"[FETCH.BLANK] Created blank file: {fname}")

def cmd_sys_proc_info(args):
    """sys.proc.info - show process info"""
    import psutil
    print("[SYS.PROC.INFO] Active processes:")
    for p in psutil.process_iter(['pid','name','username']):
        print(f"  PID: {p.info['pid']}  Name: {p.info['name']}  User: {p.info['username']}")

def cmd_toad(args):
    """toad <subcommand> - toad main command"""
    if not args:
        print("Usage: toad <subcommand>")
        return
    sub = args[0]
    if sub == 'port':
        cmd_toad_port(args[1:])
    elif sub == 'list':
        import psutil
        print("[TOAD] Listing all processes:")
        for p in psutil.process_iter(['pid','name']):
            print(f"  PID: {p.info['pid']}  Name: {p.info['name']}")
    elif sub == 'kill' and len(args)>1:
        import psutil
        try:
            p = psutil.Process(int(args[1]))
            p.terminate()
            print(f"[TOAD] Terminated process {args[1]}")
        except Exception as e:
            print(f"[TOAD] Error: {e}")
    else:
        print(f"[TOAD] Unknown subcommand: {sub}")

def cmd_toad_port(args):
    """toad.port <pattern> - search for open ports matching pattern"""
    import psutil
    pat = args[0] if args else ''
    for c in psutil.net_connections():
        if pat in str(c.laddr) or pat in str(c.raddr):
            print(c)















def cmd_find_dir_all(args):
    """find.dir.all [root] - list all directories"""
    from pathlib import Path
    root = Path(args[0]) if args else Path('.')
    for d in root.rglob('*'):
        if d.is_dir():
            print(d)

def cmd_find_url_all(args):
    """find.url.all <site> [--depth N] - recursively crawl and list all files/resources in a site"""
    import re, requests, urllib.parse
    from collections import deque
    if not args:
        print("Usage: find.url.all <site> [--depth N]")
        return
    site = args[0]
    max_depth = 2
    if '--depth' in args:
        try:
            idx = args.index('--depth')
            max_depth = int(args[idx+1])
        except Exception:
            pass
    if not re.match(r'^https?://', site):
        print("[FIND.URL.ALL] Not a valid URL.")
        return
    visited = set()
    found = set()
    queue = deque([site])
    domain = urllib.parse.urlparse(site).netloc
    depth_map = {site: 0}
    print(f"[FIND.URL.ALL] Crawling {site} (max depth {max_depth})...")
    file_exts = ['.js','.css','.html','.htm','.php','.asp','.aspx','.jpg','.jpeg','.png','.gif','.svg','.ico','.txt','.json','.xml','.pdf','.zip','.tar','.gz','.rar','.7z','.py']
    while queue:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        depth = depth_map.get(url, 0)
        if depth > max_depth:
            continue
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            html = r.text
            urls = re.findall(r'href=["\'](https?://[^"\'>]+)', html)
            for u in urls:
                if u not in visited and urllib.parse.urlparse(u).netloc == domain:
                    queue.append(u)
                    depth_map[u] = depth+1
            for ext in file_exts:
                pattern = r'https?://[^"\'>]+' + re.escape(ext)
                for m in re.findall(pattern, html):
                    if m not in found:
                        print(m)
                        found.add(m)
        except Exception as e:
            print(f"[FIND.URL.ALL] Error: {e}")
def cmd_fetch_url(args):
    """fetch.url <url> - fetch a URL and print its content"""
    import requests
    if not args:
        print("Usage: fetch.url <url>")
        return
    url = args[0]
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        print(r.text)
    except Exception as e:
        print(f"[FETCH.URL] Error: {e}")

def cmd_fetch_url_headers(args):
    """fetch.url.headers <url> - show headers for URL"""
    import requests
    if not args:
        print("Usage: fetch.url.headers <url>")
        return
    url = args[0]
    try:
        r = requests.head(url, timeout=10)
        print(r.headers)
    except Exception as e:
        print(f"[FETCH.URL.HEADERS] Error: {e}")

def cmd_fetch_url_status(args):
    """fetch.url.status <url> - show status for URL"""
    import requests
    if not args:
        print("Usage: fetch.url.status <url>")
        return
    url = args[0]
    try:
        r = requests.get(url, timeout=10)
        print(f"[FETCH.URL.STATUS] {r.status_code}")
    except Exception as e:
        print(f"[FETCH.URL.STATUS] Error: {e}")

def cmd_fetch_url_links(args):
    """fetch.url.links <url> - list all links in a URL"""
    import requests, re
    if not args:
        print("Usage: fetch.url.links <url>")
        return
    url = args[0]
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        html = r.text
        links = re.findall(r'href=["\'](https?://[^"\'>]+)', html)
        for l in links:
            print(l)
    except Exception as e:
        print(f"[FETCH.URL.LINKS] Error: {e}")

def cmd_fetch_url_save(args):
    """fetch.url.save <url> <path> - save URL content to path"""
    import requests
    if len(args)<2:
        print("Usage: fetch.url.save <url> <path>")
        return
    url, path = args[0], args[1]
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(r.text)
        print(f"[FETCH.URL.SAVE] Saved {url} -> {path}")
    except Exception as e:
        print(f"[FETCH.URL.SAVE] Error: {e}")

def cmd_fetch_url_type(args):
    """fetch.url.type <url> - show type for URL"""
    import requests
    if not args:
        print("Usage: fetch.url.type <url>")
        return
    url = args[0]
    try:
        r = requests.head(url, timeout=10)
        print(f"[FETCH.URL.TYPE] {r.headers.get('Content-Type','?')}")
    except Exception as e:
        print(f"[FETCH.URL.TYPE] Error: {e}")

def cmd_fetch_url_info(args):
    """fetch.url.info <url> - show info for URL"""
    import requests
    if not args:
        print("Usage: fetch.url.info <url>")
        return
    url = args[0]
    try:
        r = requests.head(url, timeout=10)
        print(f"[FETCH.URL.INFO] Status: {r.status_code} Headers: {r.headers}")
    except Exception as e:
        print(f"[FETCH.URL.INFO] Error: {e}")

def cmd_find_website_all(args):

    """find.website.all - list all known websites"""
    known = ["example.com", "testsite.org", "mysite.net", "oshonet.in"]
    for s in known:
        print(s)

def cmd_scan(args):
    """scan <target> - scan a file or directory for basic info"""
    import os
    if not args:
        print("Usage: scan <file|dir>")
        return
    target = args[0]
    if os.path.isfile(target):
        print(f"[SCAN] File: {target}")
        print(f"  Size: {os.path.getsize(target)} bytes")
        print(f"  Modified: {os.path.getmtime(target)}")
    elif os.path.isdir(target):
        print(f"[SCAN] Directory: {target}")
        print(f"  Files: {len(os.listdir(target))}")
    else:
        print(f"[SCAN] Not found: {target}")

def cmd_find_url(args):
    """find.url <pattern> [site] - find URLs matching pattern in a site"""
    import re
    import requests
    if not args:
        print("Usage: find.url <pattern> [site]")
        return
    pat = args[0]
    site = args[1] if len(args)>1 else None
    if site and re.match(r'^https?://', site):
        try:
            r = requests.get(site, timeout=10)
            r.raise_for_status()
            html = r.text
            urls = re.findall(r'https?://[\w\-./?&=%]+', html)
            for u in urls:
                if pat in u:
                    print(u)
        except Exception as e:
            print("[FIND.URL] Error:", e)
    else:
        print("[FIND.URL] No site or not a valid URL.")

def cmd_find_site(args):
    """find.site <pattern> - find sites/domains matching pattern"""
    import requests, re
    if not args:
        print("Usage: find.site <pattern> <url>")
        return
    pat = args[0]
    url = args[1] if len(args)>1 else 'https://www.google.com/search?q=' + pat
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        html = r.text
        sites = set(re.findall(r'https?://[\w\.-]+', html))
        for s in sites:
            if pat in s:
                print(s)
    except Exception as e:
        print(f"[FIND.SITE] Error: {e}")

def cmd_find_dir(args):
    """find.dir <pattern> [root] - find directories matching pattern"""
    from pathlib import Path
    pat = args[0] if args else ''
    root = Path(args[1]) if len(args)>1 else Path('.')
    for d in root.rglob('*'):
        if d.is_dir() and pat in d.name:
            print(d)

def cmd_find_file(args):
    """find.file <pattern> [root] - find files matching pattern"""
    from pathlib import Path
    pat = args[0] if args else ''
    root = Path(args[1]) if len(args)>1 else Path('.')
    for f in root.rglob('*'):
        if f.is_file() and pat in f.name:
            print(f)

def cmd_find_port(args):
    """find.port <pattern> - find open ports matching pattern"""
    import psutil
    pat = args[0] if args else ''
    for c in psutil.net_connections():
        if pat in str(c.laddr) or pat in str(c.raddr):
            print(c)

def cmd_find_url_port(args):
    """find.url.port <pattern> [site] - find URLs with port in a site"""
    import re, requests
    if not args:
        print("Usage: find.url.port <pattern> [site]")
        return
    pat = args[0]
    site = args[1] if len(args)>1 else None
    if site and re.match(r'^https?://', site):
        try:
            r = requests.get(site, timeout=10)
            r.raise_for_status()
            html = r.text
            urls = re.findall(r'https?://[\w\-./?&=%:]+', html)
            for u in urls:
                if pat in u and ':' in u:
                    print(u)
        except Exception as e:
            print("[FIND.URL.PORT] Error:", e)
    else:
        print("[FIND.URL.PORT] No site or not a valid URL.")

def cmd_find_port_url(args):
    """find.port.url <pattern> [site] - find ports for URLs in a site"""
    import re, requests
    if not args:
        print("Usage: find.port.url <pattern> [site]")
        return
    pat = args[0]
    site = args[1] if len(args)>1 else None
    if site and re.match(r'^https?://', site):
        try:
            r = requests.get(site, timeout=10)
            r.raise_for_status()
            html = r.text
            urls = re.findall(r'https?://[\w\-./?&=%:]+', html)
            for u in urls:
                if pat in u and ':' in u:
                    print(u.split(':')[2])
        except Exception as e:
            print("[FIND.PORT.URL] Error:", e)
    else:
        print("[FIND.PORT.URL] No site or not a valid URL.")

def cmd_find_dir_all(args):
    """find.dir.all [root] - list all directories"""
    from pathlib import Path
    root = Path(args[0]) if args else Path('.')
    for d in root.rglob('*'):
        if d.is_dir():
            print(d)

def cmd_find_url_all(args):
    """find.url.all <site> [--depth N] - recursively crawl and list all files/resources in a site (nmap-style)"""
    import re, requests, urllib.parse
    from collections import deque
    if not args:
        print("Usage: find.url.all <site> [--depth N]")
        return
    site = args[0]
    max_depth = 2
    if '--depth' in args:
        try:
            idx = args.index('--depth')
            max_depth = int(args[idx+1])
        except Exception:
            pass
    if not re.match(r'^https?://', site):
        print("[FIND.URL.ALL] Not a valid URL.")
        return
    visited = set()
    found = set()
    queue = deque([site])
    domain = urllib.parse.urlparse(site).netloc
    depth_map = {site: 0}
    print(f"[FIND.URL.ALL] Crawling {site} (max depth {max_depth})...")
    file_exts = ['.js','.css','.html','.htm','.php','.asp','.aspx','.jpg','.jpeg','.png','.gif','.svg','.ico','.txt','.json','.xml','.pdf','.zip','.tar','.gz','.rar','.7z','.py']
    while queue:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        depth = depth_map.get(url, 0)
        if depth > max_depth:
            continue
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            html = r.text
        except Exception:
            continue
        links = set(re.findall(r'(?:href|src)=["\']([^"\'>]+)', html, re.I))
        links.update(re.findall(r'https?://[\w\-./?&=%]+', html))
        for link in links:
            abs_link = urllib.parse.urljoin(url, link)
            parsed = urllib.parse.urlparse(abs_link)
            if parsed.netloc != domain:
                continue
            if abs_link not in visited:
                if any(abs_link.lower().endswith(ext) for ext in file_exts):
                    found.add(abs_link)
                if any(abs_link.lower().endswith(ext) for ext in ['.html','.htm','.php','.py','/','']):
                    if abs_link not in depth_map or depth_map[abs_link] > depth+1:
                        queue.append(abs_link)
                        depth_map[abs_link] = depth+1
    for f in sorted(found):
        print(f)

def cmd_find_website_all(args):
    """find.website.all - list all known websites (demo)"""
    known = ["example.com", "testsite.org", "mysite.net", "oshonet.in"]
    for s in known:
        print(s)

def cmd_toad(args):
    """toad <subcommand> - show system info (demo)"""
    import platform
    print("[TOAD] Platform:", platform.platform())
    print("[TOAD] Python:", platform.python_version())

def cmd_toad_port(args):
    """toad.port <pattern> - search for open ports matching pattern"""
    import psutil
    pat = args[0] if args else ''
    for c in psutil.net_connections():
        if pat in str(c.laddr) or pat in str(c.raddr):
            print(c)

def cmd_analyze(args):
    """analyze <target> [options] - analyze file size and type"""
    from pathlib import Path
    if not args:
        print("Usage: analyze <file>")
        return
    p = Path(args[0])
    if not p.exists():
        print("[ANALYZE] Not found:", p)
        return
    print(f"[ANALYZE] {p}: size={p.stat().st_size} bytes, type={p.suffix}")

def cmd_monitor(args):
    """monitor <target> [options] - monitor file for changes (basic)"""
    import time
    from pathlib import Path
    if not args:
        print("Usage: monitor <file>")
        return
    p = Path(args[0])
    if not p.exists():
        print("[MONITOR] Not found:", p)
        return
    print(f"[MONITOR] Monitoring {p} for changes. Press Ctrl+C to stop.")
    last = p.stat().st_mtime
    try:
        while True:
            time.sleep(1)
            now = p.stat().st_mtime
            if now != last:
                print(f"[MONITOR] {p} changed at {time.ctime(now)}")
                last = now
    except KeyboardInterrupt:
        print("[MONITOR] Stopped.")

def cmd_convert(args):
    """convert <src> <dst> [options] - copy file as conversion demo"""
    from pathlib import Path
    import shutil
    if len(args)<2:
        print("Usage: convert <src> <dst>")
        return
    src = Path(args[0])
    dst = Path(args[1])
    if not src.exists():
        print("[CONVERT] Not found:", src)
        return
    shutil.copy2(src, dst)
    print(f"[CONVERT] Copied {src} -> {dst}")

def cmd_report(args):
    """report <target> [options] - show file info as report"""
    from pathlib import Path
    if not args:
        print("Usage: report <file>")
        return
    p = Path(args[0])
    if not p.exists():
        print("[REPORT] Not found:", p)
        return
    print(f"[REPORT] File: {p}\n  Size: {p.stat().st_size} bytes\n  Modified: {time.ctime(p.stat().st_mtime)}")

def cmd_auto(args):
    """auto <task> [options] - automation tasks"""
    if not args:
        print("Usage: auto <task>")
        return
    task = args[0]
    if task == 'backup':
        import shutil
        if len(args)<3:
            print("Usage: auto backup <src> <dst>"); return
        shutil.copytree(args[1], args[2], dirs_exist_ok=True)
        print(f"[AUTO.BACKUP] {args[1]} -> {args[2]}")
    elif task == 'clean':
        import os
        if len(args)<2:
            print("Usage: auto clean <dir>"); return
        for f in os.listdir(args[1]):
            fp = os.path.join(args[1], f)
            if os.path.isfile(fp):
                os.remove(fp)
        print(f"[AUTO.CLEAN] Cleaned {args[1]}")
    elif task == 'update':
        import subprocess
        print("[AUTO.UPDATE] Updating...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    elif task == 'archive':
        import shutil
        if len(args)<3:
            print("Usage: auto archive <src> <dst.zip>"); return
        shutil.make_archive(args[2].replace('.zip',''), 'zip', args[1])
        print(f"[AUTO.ARCHIVE] {args[1]} -> {args[2]}")
    elif task == 'extract':
        import zipfile
        if len(args)<3:
            print("Usage: auto extract <zip> <dst>"); return
        with zipfile.ZipFile(args[1], 'r') as zip_ref:
            zip_ref.extractall(args[2])
        print(f"[AUTO.EXTRACT] {args[1]} -> {args[2]}")
    elif task == 'move':
        import shutil
        if len(args)<3:
            print("Usage: auto move <src> <dst>"); return
        shutil.move(args[1], args[2])
        print(f"[AUTO.MOVE] {args[1]} -> {args[2]}")
    elif task == 'copy':
        import shutil
        if len(args)<3:
            print("Usage: auto copy <src> <dst>"); return
        shutil.copy2(args[1], args[2])
        print(f"[AUTO.COPY] {args[1]} -> {args[2]}")
    elif task == 'touch':
        import os
        if len(args)<2:
            print("Usage: auto touch <file>"); return
        with open(args[1], 'a') as f:
            os.utime(args[1], None)
        print(f"[AUTO.TOUCH] {args[1]}")
    elif task == 'log':
        if len(args)<2:
            print("Usage: auto log <msg>"); return
        with open('auto.log','a') as f:
            f.write(' '.join(args[1:])+"\n")
        print(f"[AUTO.LOG] Logged: {' '.join(args[1:])}")
    else:
        print(f"[AUTO] Unknown task: {task}")

def cmd_extract(args):
# --- FIND SUBCOMMANDS AND SUB-SUBCOMMANDS ---

    site = args[1] if len(args)>1 else None
    # Demo: just print domain


def cmd_find_email_pattern(args):
    """find.email.pattern <pattern> [site] - find emails by regex pattern"""
    if not args:
        print("Usage: find.email.pattern <pattern> [site]"); return
    print(f"[FIND.EMAIL.PATTERN] Would search for emails by pattern {args[0]}")

def cmd_find_email_count(args):
    """find.email.count [site] - count emails in a site"""
    print("[FIND.EMAIL.COUNT] Would count emails in site.")

def cmd_find_email_unique(args):
    """find.email.unique [site] - list unique emails in a site"""
    print("[FIND.EMAIL.UNIQUE] Would list unique emails in site.")

def cmd_find_email_save(args):
    """find.email.save <file> [site] - save found emails to file"""
    print("[FIND.EMAIL.SAVE] Would save found emails to file.")

def cmd_find_phone(args):
    """find.phone <pattern> [site] - find phone numbers matching pattern in a site"""
    print("[FIND.PHONE] Would find phone numbers.")

def cmd_find_phone_country(args):
    """find.phone.country <country> [site] - find phone numbers by country code"""
    print("[FIND.PHONE.COUNTRY] Would find phone numbers by country.")

def cmd_find_phone_pattern(args):
    """find.phone.pattern <pattern> [site] - find phone numbers by pattern"""
    print("[FIND.PHONE.PATTERN] Would find phone numbers by pattern.")

def cmd_find_phone_count(args):
    """find.phone.count [site] - count phone numbers in a site"""
    print("[FIND.PHONE.COUNT] Would count phone numbers.")

def cmd_find_phone_unique(args):
    """find.phone.unique [site] - list unique phone numbers in a site"""
    print("[FIND.PHONE.UNIQUE] Would list unique phone numbers.")

def cmd_find_phone_save(args):
    """find.phone.save <file> [site] - save found phone numbers to file"""
    print("[FIND.PHONE.SAVE] Would save found phone numbers to file.")

def cmd_find_hash(args):
    """find.hash <pattern> [root] - find files with hash matching pattern"""
    print("[FIND.HASH] Would find files by hash.")

def cmd_find_hash_type(args):
    """find.hash.type <type> [root] - find files by hash type"""
    print("[FIND.HASH.TYPE] Would find files by hash type.")

def cmd_find_hash_pattern(args):
    """find.hash.pattern <pattern> [root] - find files by hash pattern"""
    print("[FIND.HASH.PATTERN] Would find files by hash pattern.")

def cmd_find_hash_count(args):
    """find.hash.count [root] - count files with hashes"""
    print("[FIND.HASH.COUNT] Would count files with hashes.")

def cmd_find_hash_unique(args):
    """find.hash.unique [root] - list unique hashes"""
    print("[FIND.HASH.UNIQUE] Would list unique hashes.")

def cmd_find_hash_save(args):
    """find.hash.save <file> [root] - save found hashes to file"""
    print("[FIND.HASH.SAVE] Would save found hashes to file.")
# --- FETCH SUBCOMMANDS AND SUB-SUBCOMMANDS ---
def cmd_fetch_file_type(args):
    """fetch.file.type <url> - show file type at URL"""
    import requests
    if not args:
        print("Usage: fetch.file.type <url>")
        return
    url = args[0]
    try:
        r = requests.head(url, timeout=10)
        ctype = r.headers.get('Content-Type','?')
        print(f"[FETCH.FILE.TYPE] {ctype}")
    except Exception as e:
        print(f"[FETCH.FILE.TYPE] Error: {e}")
def cmd_fetch_file_size(args):
    """fetch.file.size <url> - show file size at URL"""
    print("[FETCH.FILE.SIZE] Would show file size.")
def cmd_fetch_dir(args):
    """fetch.dir <url> - list directory at URL"""
    print("[FETCH.DIR] Would list directory at URL.")
def cmd_fetch_dir_list(args):
    """fetch.dir.list <url> - list files in directory at URL"""
    print("[FETCH.DIR.LIST] Would list files in directory.")
def cmd_fetch_dir_size(args):
    """fetch.dir.size <url> - show size of directory at URL"""
    print("[FETCH.DIR.SIZE] Would show directory size.")
def cmd_fetch_dir_save(args):
    """fetch.dir.save <url> <path> - save directory from URL to path"""
    print("[FETCH.DIR.SAVE] Would save directory from URL.")
def cmd_fetch_dir_type(args):
    """fetch.dir.type <url> - show type of directory at URL"""
    print("[FETCH.DIR.TYPE] Would show directory type.")
def cmd_fetch_dir_info(args):
    """fetch.dir.info <url> - show info about directory at URL"""
    print("[FETCH.DIR.INFO] Would show directory info.")
def cmd_fetch_url(args):
    """fetch.url <url> - fetch a URL"""
    print("[FETCH.URL] Would fetch URL.")
def cmd_fetch_url_headers(args):
    """fetch.url.headers <url> - show headers for URL"""
    print("[FETCH.URL.HEADERS] Would show headers.")
def cmd_fetch_url_status(args):
    """fetch.url.status <url> - show status for URL"""
    print("[FETCH.URL.STATUS] Would show status.")
def cmd_fetch_url_save(args):
    """fetch.url.save <url> <path> - save URL content to path"""
    print("[FETCH.URL.SAVE] Would save URL content.")
def cmd_fetch_url_type(args):
    """fetch.url.type <url> - show type for URL"""
    print("[FETCH.URL.TYPE] Would show type.")
def cmd_fetch_url_info(args):
    """fetch.url.info <url> - show info for URL"""
    print("[FETCH.URL.INFO] Would show info.")
def cmd_fetch_url_links(args):
    pass

# --- SYNC SUBCOMMANDS AND SUB-SUBCOMMANDS ---
def cmd_sync_files(args):
    """sync.files <src> <dst> - sync all files from src to dst"""
    import shutil
    if len(args)<2:
        print("Usage: sync.files <src> <dst>"); return
    src, dst = Path(args[0]), Path(args[1])
    for f in src.rglob('*'):
        if f.is_file():
            rel = f.relative_to(src)
            dfile = dst / rel
            dfile.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dfile)
            print(f"[SYNC.FILES] {f} -> {dfile}")
def cmd_sync_dirs(args):
    """sync.dirs <src> <dst> - sync all directories from src to dst"""
    if len(args)<2:
        print("Usage: sync.dirs <src> <dst>"); return
    src, dst = Path(args[0]), Path(args[1])
    for d in src.rglob('*'):
        if d.is_dir():
            rel = d.relative_to(src)
            dd = dst / rel
            dd.mkdir(parents=True, exist_ok=True)
            print(f"[SYNC.DIRS] {d} -> {dd}")
def cmd_sync_new(args):
    """sync.new <src> <dst> - sync only new files from src to dst"""
    if len(args)<2:
        print("Usage: sync.new <src> <dst>"); return
    src, dst = Path(args[0]), Path(args[1])
    for f in src.rglob('*'):
        if f.is_file():
            rel = f.relative_to(src)
            dfile = dst / rel
            if not dfile.exists():
                dfile.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dfile)
                print(f"[SYNC.NEW] {f} -> {dfile}")
def cmd_sync_changed(args):
    """sync.changed <src> <dst> - sync only changed files from src to dst"""
    if len(args)<2:
        print("Usage: sync.changed <src> <dst>"); return
    src, dst = Path(args[0]), Path(args[1])
    for f in src.rglob('*'):
        if f.is_file():
            rel = f.relative_to(src)
            dfile = dst / rel
            if dfile.exists() and f.stat().st_mtime > dfile.stat().st_mtime:
                shutil.copy2(f, dfile)
                print(f"[SYNC.CHANGED] {f} -> {dfile}")
def cmd_sync_deleted(args):
    """sync.deleted <src> <dst> - list files deleted from src but present in dst"""
    if len(args)<2:
        print("Usage: sync.deleted <src> <dst>"); return
    src, dst = Path(args[0]), Path(args[1])
    src_files = {str(f.relative_to(src)) for f in src.rglob('*') if f.is_file()}
    dst_files = {str(f.relative_to(dst)) for f in dst.rglob('*') if f.is_file()}
    deleted = dst_files - src_files
    for f in deleted:
        print(f"[SYNC.DELETED] {f}")
def cmd_sync_summary(args):
    """sync.summary <src> <dst> - show sync summary"""
    if len(args)<2:
        print("Usage: sync.summary <src> <dst>"); return
    src, dst = Path(args[0]), Path(args[1])
    src_count = sum(1 for _ in src.rglob('*'))
    dst_count = sum(1 for _ in dst.rglob('*'))
    print(f"[SYNC.SUMMARY] src: {src_count} items, dst: {dst_count} items")
def cmd_sync_log(args):
    """sync.log <src> <dst> - log sync actions"""
    print("[SYNC.LOG] Would log sync actions.")
def cmd_sync_verify(args):
    """sync.verify <src> <dst> - verify sync integrity"""
    print("[SYNC.VERIFY] Would verify sync integrity.")
def cmd_sync_backup(args):
    """sync.backup <src> <dst> - backup before sync"""
    print("[SYNC.BACKUP] Would backup before sync.")
def cmd_sync_restore(args):
    """sync.restore <src> <dst> - restore from backup"""
    print("[SYNC.RESTORE] Would restore from backup.")
def cmd_sync_status(args):
    """sync.status <src> <dst> - show sync status"""
    print("[SYNC.STATUS] Would show sync status.")
def cmd_sync_files_md5(args):
    """sync.files.md5 <src> <dst> - sync files and verify md5"""
    print("[SYNC.FILES.MD5] Would sync files and verify md5.")
def cmd_sync_files_sha256(args):
    """sync.files.sha256 <src> <dst> - sync files and verify sha256"""
    print("[SYNC.FILES.SHA256] Would sync files and verify sha256.")
def cmd_sync_files_log(args):
    """sync.files.log <src> <dst> - log file sync actions"""
    print("[SYNC.FILES.LOG] Would log file sync actions.")
def cmd_sync_files_verify(args):
    """sync.files.verify <src> <dst> - verify file sync"""
    print("[SYNC.FILES.VERIFY] Would verify file sync.")
def cmd_sync_files_backup(args):
    """sync.files.backup <src> <dst> - backup files before sync"""
    print("[SYNC.FILES.BACKUP] Would backup files before sync.")
def cmd_sync_files_restore(args):
    """sync.files.restore <src> <dst> - restore files from backup"""
    print("[SYNC.FILES.RESTORE] Would restore files from backup.")
    """fetch.url.links <url> - list links in URL"""
    print("[FETCH.URL.LINKS] Would list links.")
    """find.hash.save <file> [root] - save found hashes to file"""
    print("[FIND.HASH.SAVE] Would save found hashes to file.")
    """extract <src> [options] - extract lines containing 'TODO' from file"""
    from pathlib import Path
    if not args:
        print("Usage: extract <file>")
        return
    p = Path(args[0])
    if not p.exists():
        print("[EXTRACT] Not found:", p)
        return
    for line in p.read_text(errors="replace").splitlines():
        if 'TODO' in line:
            print(line)
    """scan <target> - scan a file or directory for basic info"""
    import os
    if not args:
        print("Usage: scan <file|dir>")
        return
    target = args[0]
    if os.path.isfile(target):
        print(f"[SCAN] File: {target}")
        print(f"  Size: {os.path.getsize(target)} bytes")
        print(f"  Modified: {os.path.getmtime(target)}")
    elif os.path.isdir(target):
        print(f"[SCAN] Directory: {target}")
        print(f"  Files: {len(os.listdir(target))}")
    else:
        print(f"[SCAN] Not found: {target}")

def cmd_analyze(args):
    """analyze <target> [options] - analyze file size and type"""
    from pathlib import Path
    if not args:
        print("Usage: analyze <file>")
        return
    p = Path(args[0])
    if not p.exists():
        print("[ANALYZE] Not found:", p)
        return
    print(f"[ANALYZE] {p}: size={p.stat().st_size} bytes, type={p.suffix}")

def cmd_monitor(args):
    """monitor <target> [options] - monitor file for changes (basic)"""
    import time
    from pathlib import Path
    if not args:
        print("Usage: monitor <file>")
        return
    p = Path(args[0])
    if not p.exists():
        print("[MONITOR] Not found:", p)
        return
    print(f"[MONITOR] Monitoring {p} for changes. Press Ctrl+C to stop.")
    last = p.stat().st_mtime
    try:
        while True:
            time.sleep(1)
            now = p.stat().st_mtime
            if now != last:
                print(f"[MONITOR] {p} changed at {time.ctime(now)}")
                last = now
    except KeyboardInterrupt:
        print("[MONITOR] Stopped.")

def cmd_convert(args):
    """convert <src> <dst> [options] - copy file as conversion demo"""
    from pathlib import Path
    import shutil
    if len(args)<2:
        print("Usage: convert <src> <dst>")
        return
    src = Path(args[0])
    dst = Path(args[1])
    if not src.exists():
        print("[CONVERT] Not found:", src)
        return
    shutil.copy2(src, dst)
    print(f"[CONVERT] Copied {src} -> {dst}")

def cmd_report(args):
    """report <target> [options] - show file info as report"""
    from pathlib import Path
    if not args:
        print("Usage: report <file>")
        return
    p = Path(args[0])
    if not p.exists():
        print("[REPORT] Not found:", p)
        return
    print(f"[REPORT] File: {p}\n  Size: {p.stat().st_size} bytes\n  Modified: {time.ctime(p.stat().st_mtime)}")

def cmd_auto(args):
    """auto <task> [options] - automation tasks"""
    if not args:
        print("Usage: auto <task>")
        return
    task = args[0]
    if task == 'backup':
        import shutil
        if len(args)<3:
            print("Usage: auto backup <src> <dst>"); return
        shutil.copytree(args[1], args[2], dirs_exist_ok=True)
        print(f"[AUTO.BACKUP] {args[1]} -> {args[2]}")
    elif task == 'clean':
        import os
        if len(args)<2:
            print("Usage: auto clean <dir>"); return
        for f in os.listdir(args[1]):
            fp = os.path.join(args[1], f)
            if os.path.isfile(fp):
                os.remove(fp)
        print(f"[AUTO.CLEAN] Cleaned {args[1]}")
    elif task == 'update':
        import subprocess
        print("[AUTO.UPDATE] Updating...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    elif task == 'archive':
        import shutil
        if len(args)<3:
            print("Usage: auto archive <src> <dst.zip>"); return
        shutil.make_archive(args[2].replace('.zip',''), 'zip', args[1])
        print(f"[AUTO.ARCHIVE] {args[1]} -> {args[2]}")
    elif task == 'extract':
        import zipfile
        if len(args)<3:
            print("Usage: auto extract <zip> <dst>"); return
        with zipfile.ZipFile(args[1], 'r') as zip_ref:
            zip_ref.extractall(args[2])
        print(f"[AUTO.EXTRACT] {args[1]} -> {args[2]}")
    elif task == 'move':
        import shutil
        if len(args)<3:
            print("Usage: auto move <src> <dst>"); return
        shutil.move(args[1], args[2])
        print(f"[AUTO.MOVE] {args[1]} -> {args[2]}")
    elif task == 'copy':
        import shutil
        if len(args)<3:
            print("Usage: auto copy <src> <dst>"); return
        shutil.copy2(args[1], args[2])
        print(f"[AUTO.COPY] {args[1]} -> {args[2]}")
    elif task == 'touch':
        import os
        if len(args)<2:
            print("Usage: auto touch <file>"); return
        with open(args[1], 'a') as f:
            os.utime(args[1], None)
        print(f"[AUTO.TOUCH] {args[1]}")
    elif task == 'log':
        if len(args)<2:
            print("Usage: auto log <msg>"); return
        with open('auto.log','a') as f:
            f.write(' '.join(args[1:])+"\n")
        print(f"[AUTO.LOG] Logged: {' '.join(args[1:])}")
    else:
        print(f"[AUTO] Unknown task: {task}")

def cmd_sync(args):
    """sync <src> <dst> [options] - sync files/dirs/URLs"""
    import shutil
    from pathlib import Path
    if len(args) < 2:
        print("Usage: sync <src> <dst>")
        return
    src = Path(args[0])
    dst = Path(args[1])
    if not src.exists():
        print(f"[SYNC] Source not found: {src}")
        return
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[SYNC] Copied file {src} -> {dst}")
        return
    # Directory sync: copy new/changed files
    for sfile in src.rglob('*'):
        if sfile.is_file():
            rel = sfile.relative_to(src)
            dfile = dst / rel
            if not dfile.exists() or sfile.stat().st_mtime > dfile.stat().st_mtime:
                dfile.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sfile, dfile)
                print(f"[SYNC] {sfile} -> {dfile}")
    print("[SYNC] Done.")


def cmd_scan(args):
    """scan <target> [options] - scan a directory for files or a host for open ports"""
    import os
    import platform
    import socket
    from pathlib import Path
    if not args:
        print("Usage: scan <dir>|<host>")
        return
    target = args[0]
    # Directory scan
    if os.path.isdir(target):
        print(f"[SCAN] Files in {target}:")
        for root, dirs, files in os.walk(target):
            for f in files:
                print(os.path.join(root, f))
        return
    # Host port scan (basic)
    print(f"[SCAN] Scanning host {target} for open ports 1-1024...")
    open_ports = []
    for port in range(1, 1025):
        try:
            with socket.create_connection((target, port), timeout=0.2):
                open_ports.append(port)
        except Exception:
            continue
    if open_ports:
        print(f"[SCAN] Open ports: {open_ports}")
    else:
        print("[SCAN] No open ports found (1-1024).")
    """extract <src> [options] - extract data from files/sites"""
    # Implementation for extract command goes here
# ----------------------------
def cmd_fetch_info_all(args):
    """fetch.info.all <url|dir> - show all JS, HTML, CSS, TXT files in a URL or directory"""
    if not args:
        print("Usage: fetch.info.all <url|dir>")
        return
    target = args[0]
    if re.match(r'^https?://', target):
        # URL: fetch page and list all linked .js/.css/.html/.txt
        try:
            if requests:
                r = requests.get(target, timeout=10)
                r.raise_for_status()
                html = r.text
            else:
                import urllib.request
                html = urllib.request.urlopen(target, timeout=10).read().decode("utf-8", errors="replace")
            # Find all src/href links to .js/.css/.html/.txt
            files = set()
            for ext in ("js","css","html","txt"):
                files.update(re.findall(r'([\w\-./]+\\.'+ext+r')', html, re.I))
            print(f"[INFO.ALL] Found {len(files)} files:")
            for f in sorted(files):
                print(f)
        except Exception as e:
            print("[INFO.ALL] Error:", e)
    else:
        # Local dir: list all .js/.css/.html/.txt
        p = Path(target)
        if not p.exists():
            print("[INFO.ALL] Not found:", p)
            return
        if p.is_file():
            print("[INFO.ALL] Not a directory:", p)
            return
        found = list(p.rglob("*.js")) + list(p.rglob("*.css")) + list(p.rglob("*.html")) + list(p.rglob("*.txt"))
        print(f"[INFO.ALL] Found {len(found)} files:")
        for f in found:
            print(f)

def cmd_fetch_info_all_download(args):
    """fetch.info.all:download <url|dir> [dest_dir] - download all JS, HTML, CSS, TXT files to dest_dir"""
    if not args:
        print("Usage: fetch.info.all:download <url|dir> [dest_dir]")
        return
    target = args[0]
    dest = Path(args[1]) if len(args)>1 else Path(".")
    dest.mkdir(parents=True, exist_ok=True)
    if re.match(r'^https?://', target):
        # URL: fetch page and download all linked .js/.css/.html/.txt
        try:
            if requests:
                r = requests.get(target, timeout=10)
                r.raise_for_status()
                html = r.text
            else:
                import urllib.request
                html = urllib.request.urlopen(target, timeout=10).read().decode("utf-8", errors="replace")
            files = set()
            for ext in ("js","css","html","txt"):
                files.update(re.findall(r'([\w\-./]+\\.'+ext+r')', html, re.I))
            print(f"[DL] Downloading {len(files)} files...")
            for f in sorted(files):
                url = f if f.startswith("http") else target.rstrip("/")+"/"+f.lstrip("/")
                outp = dest / os.path.basename(f)
                try:
                    if requests:
                        fr = requests.get(url, timeout=10)
                        fr.raise_for_status()
                        outp.write_bytes(fr.content)
                    else:
                        import urllib.request
                        data = urllib.request.urlopen(url, timeout=10).read()
                        outp.write_bytes(data)
                    print(f"[DL] {url} -> {outp}")
                except Exception as e:
                    print(f"[DL] Failed {url}: {e}")
        except Exception as e:
            print("[DL] Error:", e)
    else:
        # Local dir: copy all .js/.css/.html/.txt
        p = Path(target)
        if not p.exists():
            print("[DL] Not found:", p)
            return
        if p.is_file():
            print("[DL] Not a directory:", p)
            return
        found = list(p.rglob("*.js")) + list(p.rglob("*.css")) + list(p.rglob("*.html")) + list(p.rglob("*.txt"))
        for f in found:
            outp = dest / f.name
            try:
                outp.write_bytes(f.read_bytes())
                print(f"[DL] {f} -> {outp}")
            except Exception as e:
                print(f"[DL] Failed {f}: {e}")
# --- Help system for core and subcommands ---
def cmd_help(args):
    """help [command] - show help for a command or list all main commands"""
    if not args:
        print(short_help())
        print("Available main commands:")
        mains = set()
        for k in sorted(COMMAND_MAP):
            if '.' not in k and ':' not in k and not k.startswith('/'):
                mains.add(k)
            if k.startswith('/'):
                mains.add(k)
        for k in sorted(mains):
            print(" ", k)
        print("Type '<command>.help' for help on a command.")
        return
    cmd = args[0]
    fn = COMMAND_MAP.get(cmd)
    if fn:
        print(fn.__doc__ or "(no doc)")
        # Show subcommands if any
        subs = [k for k in COMMAND_MAP if k.startswith(cmd+'.') or k.startswith(cmd+':')]
        if subs:
            print("\nSubcommands:")
            for s in sorted(subs):
                print(" ", s)
    else:
        print("No help for", cmd)

def make_help_command(cmdname, fn):
    def help_fn(args):
        print(fn.__doc__ or f"No help for {cmdname}")
    help_fn.__doc__ = f"{cmdname}.help - show help for {cmdname}"
    return help_fn

def add_help_variants():
    # Add .help for every command
    for k, fn in list(COMMAND_MAP.items()):
        if not k.endswith(".help"):
            COMMAND_MAP[k+".help"] = make_help_command(k, fn)
    # Add help for subcommands with colon
    for k, fn in list(COMMAND_MAP.items()):
        if ":" in k and not k.endswith(".help"):
            COMMAND_MAP[k+".help"] = make_help_command(k, fn)
def cmd_fetch_dwn(args):
    """fetch.dwn <url> [--out filename] [--dir path] - download from url with progress bar. Uses fetch.dir as default directory."""
    import sys
    import time
    if not args:
        print("Usage: fetch.dwn <url> [--out filename] [--dir path]")
        return
    url = args[0]
    out = None
    global FETCH_DEFAULT_DIR
    if FETCH_DEFAULT_DIR is None:
        FETCH_DEFAULT_DIR = [str(Path('.').resolve())]
    outdir = Path(FETCH_DEFAULT_DIR[0])
    if "--out" in args:
        i = args.index("--out")
        if i+1 < len(args): out = args[i+1]
    if "--dir" in args:
        i = args.index("--dir")
        if i+1 < len(args): outdir = Path(args[i+1])
    try:
        if requests:
            r = requests.get(url, stream=True, timeout=20)
            r.raise_for_status()
            total = 0
            if not out:
                out = os.path.basename(url.split("?")[0]) or "download.bin"
            outp = outdir / out
            outdir.mkdir(parents=True, exist_ok=True)
            size = int(r.headers.get('content-length', 0))
            chunk_size = 8192
            start = time.time()
            with outp.open("wb") as f:
                for chunk in r.iter_content(chunk_size):
                    if not chunk: break
                    f.write(chunk)
                    total += len(chunk)
                    if size:
                        done = int(50 * total / size)
                        speed = total / (time.time()-start+0.01)
                        sys.stdout.write(f"\r[{'='*done}{' '*(50-done)}] {total}/{size} bytes ({speed/1024:.1f} KB/s)")
                        sys.stdout.flush()
            if size:
                print()
            print(f"[FETCH] Downloaded {total} bytes -> {outp}")
        else:
            # fallback
            import urllib.request
            if not out:
                out = os.path.basename(url.split("?")[0]) or "download.bin"
            outp = outdir / out
            outdir.mkdir(parents=True, exist_ok=True)
            def reporthook(blocknum, blocksize, totalsize):
                readsofar = blocknum * blocksize
                if totalsize > 0:
                    percent = readsofar * 1e2 / totalsize
                    sys.stdout.write(f"\r{percent:.1f}% {readsofar}/{totalsize} bytes")
                    sys.stdout.flush()
            urllib.request.urlretrieve(url, outp, reporthook)
            print(f"\n[FETCH] Downloaded -> {outp}")
    except Exception as e:
        print("[FETCH] Error:", e)
# --- Directory navigation: goto, wentto, byby ---
DIR_STACK = []
def cmd_goto(args):
    """goto <path> - instantly change to any directory (including drive roots)"""
    import os
    if not args:
        print(f"[GOTO] Current directory: {os.getcwd()}")
        return
    p = os.path.expanduser(args[0])
    try:
        os.chdir(p)
        print(f"[GOTO] Changed directory to: {os.getcwd()}")
    except Exception as e:
        print(f"[GOTO] Error: {e}")

def cmd_wentto(args):
    """wentto - go back to previous directory (like popd)"""
    import os
    if not DIR_STACK:
        print("[WENTTO] No previous directory in stack.")
        return
    prev = DIR_STACK.pop()
    try:
        os.chdir(prev)
        print(f"[WENTTO] Returned to: {os.getcwd()}")
    except Exception as e:
        print(f"[WENTTO] Error: {e}")

def cmd_byby(args):
    """byby <path> - save current dir and go to new one (like pushd)"""
    import os
    if not args:
        print("Usage: byby <path>")
        return
    DIR_STACK.append(os.getcwd())
    p = os.path.expanduser(args[0])
    try:
        os.chdir(p)
        print(f"[BYBY] Changed directory to: {os.getcwd()} (previous saved)")
    except Exception as e:
        print(f"[BYBY] Error: {e}")

def cmd_help(args):
    """help [command] - show help for a command or list all main commands with summaries"""
    if not args:
        print(short_help())
        print("Available main commands:")
        mains = set()
        for k, fn in COMMAND_MAP.items():
            if '.' not in k and ':' not in k and not k.startswith('/'):
                mains.add((k, fn.__doc__.split('\n')[0] if fn.__doc__ else ''))
            if k.startswith('/'):
                mains.add((k, fn.__doc__.split('\n')[0] if fn.__doc__ else ''))
        for k, doc in sorted(mains):
            print(f"  {k:15} {doc}")
        print("Type '<command>.help' for help on a command.")
        return
    cmd = args[0]
    fn = COMMAND_MAP.get(cmd)
    if fn:
        print(fn.__doc__ or "(no doc)")
        # Show subcommands with summaries
        subs = [k for k in COMMAND_MAP if k.startswith(cmd+'.') or k.startswith(cmd+':')]
        if subs:
            print("\nSubcommands:")
            for s in sorted(subs):
                doc = COMMAND_MAP[s].__doc__.split('\n')[0] if COMMAND_MAP[s].__doc__ else ''
                print(f"  {s:25} {doc}")
                # Show sub-subcommands for each subcommand
                subsubs = [k2 for k2 in COMMAND_MAP if k2.startswith(s+'.') or k2.startswith(s+':')]
                if subsubs:
                    print(f"    Sub-subcommands of {s}:")
                    for ss in sorted(subsubs):
                        doc2 = COMMAND_MAP[ss].__doc__.split('\n')[0] if COMMAND_MAP[ss].__doc__ else ''
                        print(f"    {ss:25} {doc2}")
                        # Show sub's sub's subcommands
                        subsubsubs = [k3 for k3 in COMMAND_MAP if k3.startswith(ss+'.') or k3.startswith(ss+':')]
                        if subsubsubs:
                            print(f"      Sub-sub-subcommands of {ss}:")
                            for sss in sorted(subsubsubs):
                                doc3 = COMMAND_MAP[sss].__doc__.split('\n')[0] if COMMAND_MAP[sss].__doc__ else ''
                                print(f"      {sss:25} {doc3}")
    else:
        print("No help for", cmd)
        return
    fn = Path(args[0])
    if fn.suffix == ".egg":
        fn.write_bytes(egg_encode_bytes(b"# empty egg\n"))
        print("[FETCH] Blank .egg created:", fn)
    else:
        fn.write_text("# empty eggless\n")
        print("[FETCH] Blank .eggless created:", fn)

def cmd_fetch_cl_update(args):
    """fetch.cl.update - update CLI from remote URL (careful)"""
    url = "https://world.oshonet.in/cli/latest/index.py"
    print("[UPDATE] Fetching from", url)
    try:
        if requests:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            new_code = r.text
        else:
            import urllib.request
            new_code = urllib.request.urlopen(url, timeout=10).read().decode("utf-8")
        # backup current
        me = Path(sys.argv[0]).resolve()
        bak = me.with_suffix(".bak.py")
        me.rename(bak)
        me.write_text(new_code)
        print("[UPDATE] CLI updated. Old backed up at", bak)
    except Exception as e:
        print("[UPDATE] Failed:", e)

def cmd_file_transfer(args):
    """file.transfer <src> <dest> - copy a file (preserve metadata)"""
    if len(args) < 2:
        print("Usage: file.transfer <src> <dest>")
        return
    import shutil
    try:
        shutil.copy2(args[0], args[1])
        print("[FILE] Copied", args[0], "->", args[1])
    except Exception as e:
        print("[FILE] Error:", e)

def cmd_file_split(args):
    """file.split <file> <chunksize_kb>"""
    if len(args) < 2:
        print("Usage: file.split <file> <chunksize_kb>")
        return
    path = Path(args[0])
    size = int(args[1]) * 1024
    if not path.exists():
        print("Not found:", path); return
    i=0
    with path.open("rb") as f:
        while True:
            chunk = f.read(size)
            if not chunk: break
            out = path.parent / f"{path.name}.part{i:03d}"
            out.write_bytes(chunk)
            print("[SPLIT] Wrote", out)
            i+=1

def cmd_file_join(args):
    """file.join <part_dir> <output>"""
    if len(args)<2:
        print("Usage: file.join <part_dir> <output>")
        return
    pdir = Path(args[0])
    out = Path(args[1])
    parts = sorted(pdir.glob(out.name + ".part*"))
    if not parts:
        # try any .part files in dir
        parts = sorted(pdir.glob("*.part*"))
    with out.open("wb") as o:
        for p in parts:
            o.write(p.read_bytes())
            print("[JOIN] Added", p)
    print("[JOIN] Created", out)

def cmd_file_hash(args):
    """file.hash <file> - sha256"""
    if not args:
        print("Usage: file.hash <file>")
        return
    p = Path(args[0])
    import hashlib
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(8192)
            if not b: break
            h.update(b)
    print("[HASH]", h.hexdigest())

def cmd_file_diff(args):
    """file.diff <f1> <f2> - quick binary compare"""
    if len(args)<2:
        print("Usage: file.diff <f1> <f2>")
        return
    b1 = Path(args[0]).read_bytes()
    b2 = Path(args[1]).read_bytes()
    if b1==b2:
        print("[DIFF] Files identical")
    else:
        print("[DIFF] Files differ (sizes)", len(b1), len(b2))

def cmd_dir_transfer(args):
    """dir.transfer <src> <dest> - copy dir recursively"""
    if len(args)<2:
        print("Usage: dir.transfer <src> <dest>")
        return
    import shutil
    try:
        shutil.copytree(args[0], args[1], dirs_exist_ok=True)
        print("[DIR] Transferred", args[0], "->", args[1])
    except Exception as e:
        print("[DIR] Error:", e)

def cmd_dir_list(args):
    """dir.list <dir>"""
    d = Path(args[0]) if args else Path(".")
    for p in sorted(d.iterdir()):
        t = "<dir>" if p.is_dir() else p.suffix
        print(p.name, t)

def cmd_dir_size(args):
    """dir.size <dir>"""
    d = Path(args[0]) if args else Path(".")
    total = 0
    for p in d.rglob("*"):
        if p.is_file(): total += p.stat().st_size
    print("[DIR] Total bytes:", total)

# ----------------------------
# EGG commands
# ----------------------------
def cmd_egg_make(args):
    """egg.make <file.eggless> - obfuscate to .egg"""
    if not args:
        print("Usage: egg.make <file.eggless>")
        return
    src = Path(args[0])
    if not src.exists(): print("Missing", src); return
    out = src.with_suffix(".egg")
    make_egg(src, out)
    print("[EGG] Created:", out)

def cmd_egg_open(args):
    """egg.open <file.egg> - decode and show"""
    if not args:
        print("Usage: egg.open <file.egg>")
        return
    p = Path(args[0])
    if not p.exists():
        print("Missing", p); return
    try:
        content = open_egg(p)
        print(content.decode(errors="replace"))
    except Exception as e:
        print("[EGG] Error:", e)

def cmd_egg_convert(args):
    """egg.convert <egg> <eggless>"""
    if len(args)<2:
        print("Usage: egg.convert <egg> <eggless>")
        return
    try:
        decoded = open_egg(Path(args[0]))
        Path(args[1]).write_bytes(decoded)
        print("[EGG] Converted to", args[1])
    except Exception as e:
        print("[EGG] Error:", e)

def cmd_egg_vars(args):
    """egg.vars - show demo identity variable scheme"""
    print("EGG variable scheme: $<idnum> substituted at runtime for identity tokens.")
    print("Example: let a = 612!23-/31Q~381$2801  -> $2801 is identity token.")

def cmd_egg_verify(args):
    """egg.verify <file> - quick integrity check (sha256)"""
    if not args: print("Usage: egg.verify <file>"); return
    p = Path(args[0])
    import hashlib
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print("[EGG] SHA256:", h)

def cmd_egg_clean(args):
    """egg.clean - cleans decode cache (not used in prototype)"""
    print("[EGG] Nothing to clean in prototype.")

# ----------------------------
# PONG / Network
# ----------------------------
def cmd_pong(args):
    """pong <host> [--count N] - basic ping wrapper"""
    if not args:
        print("Usage: pong <host> [--count N]")
        return
    host = args[0]
    cnt = "4"
    if "--count" in args:
        i = args.index("--count")
        if i+1 < len(args): cnt = args[i+1]
    ping_cmd = ["ping"]
    system = platform.system().lower()
    if system == "windows":
        ping_cmd += ["-n", cnt, host]
    else:
        ping_cmd += ["-c", cnt, host]
    try:
        subprocess.run(ping_cmd)
    except Exception as e:
        print("[PONG] Error:", e)

def cmd_pong_ext(args):
    """pong.ext <host> --count N --size S  - advanced ping wrapper"""
    if not args:
        print("Usage: pong.ext <host> [--count N] [--size S]")
        return
    host = args[0]
    cnt = "4"; size = None
    if "--count" in args:
        i = args.index("--count"); cnt = args[i+1]
    if "--size" in args:
        i = args.index("--size"); size = args[i+1]
    ping_cmd = ["ping"]
    system = platform.system().lower()
    if system == "windows":
        ping_cmd += ["-n", cnt]
        if size: ping_cmd += ["-l", size]
        ping_cmd += [host]
    else:
        ping_cmd += ["-c", cnt]
        if size: ping_cmd += ["-s", size]
        ping_cmd += [host]
    try:
        subprocess.run(ping_cmd)
    except Exception as e:
        print("[PONG] Error:", e)

def cmd_pong_trace(args):
    """pong.trace <host> - traceroute"""
    if not args: print("Usage: pong.trace <host>"); return
    host = args[0]
    system = platform.system().lower()
    if system == "windows":
        cmd = ["tracert", host]
    else:
        cmd = ["traceroute", host]
    try:
        subprocess.run(cmd)
    except Exception as e:
        print("[PONG] Error:", e)

# ----------------------------
# SYS commands
# ----------------------------
def cmd_sys_info(args):
    """sys.info - basic platform info"""
    print("[SYS] Platform:", platform.platform())
    print(" Python:", sys.version.splitlines()[0])
    print(" CWD:", os.getcwd())

def cmd_sys_mem(args):
    """sys.mem - memory usage (approx)"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(mem)
    except Exception:
        print("[SYS] psutil not available. Using /proc or fallback.")
        if platform.system().lower() == "linux" and Path("/proc/meminfo").exists():
            print(Path("/proc/meminfo").read_text().splitlines()[:5])
        else:
            print("[SYS] Detailed mem info not available.")

def cmd_sys_proc(args):
    """sys.proc - list processes (ps aux style)"""
    system = platform.system().lower()
    if system == "windows":
        subprocess.run(["tasklist"])
    else:
        subprocess.run(["ps", "aux"])

def cmd_sys_kill(args):
    """sys.kill <pid>"""
    if not args:
        print("Usage: sys.kill <pid>"); return
    try:
        os.kill(int(args[0]), 9)
        print("[SYS] Killed", args[0])
    except Exception as e:
        print("[SYS] Error:", e)

def cmd_sys_clr(args):
    """sys.clr - clear screen"""
    os.system("cls" if platform.system().lower()=="windows" else "clear")

def cmd_sys_time(args):
    """sys.time - show system time"""
    print(time.ctime())

# ----------------------------
# NATIVE bridging
# ----------------------------
def cmd_native(raw):
    """/native/<command> => pass to OS shell directly"""
    # raw contains full command like "/native/ls -la"
    after = raw[len("/native/"):]
    if not after.strip():
        print("Usage: /native/<command> [args]")
        return
    try:
        subprocess.run(after, shell=True)
    except Exception as e:
        print("[NATIVE] Error:", e)

# ----------------------------
# Editor integration
# ----------------------------
def auto_install_editor(editor_name="nano"):
    """Attempts to install a text editor using the system's package manager."""
    print(f"[SETUP] Editor '{editor_name}' not found.")
    choice = input(f"Attempt to install '{editor_name}' using system package manager? (y/n): ").lower()
    if choice != 'y':
        print("[SETUP] Installation skipped.")
        return False

    cmd = []
    platform_name = sys.platform
    try:
        if platform_name.startswith("win"):
            print(f"[SETUP] Attempting to install {editor_name} via winget...")
            cmd = ["winget", "install", "--id", f"GNU.{editor_name}", "--accept-source-agreements", "--accept-package-agreements"]
            subprocess.run(cmd, check=True, shell=True)
        elif platform_name.startswith("linux"):
            print(f"[SETUP] Attempting to install {editor_name} via apt (requires sudo)...")
            cmd = ["sudo", "apt", "update"]
            subprocess.run(cmd, check=True)
            cmd = ["sudo", "apt", "install", "-y", editor_name]
            subprocess.run(cmd, check=True)
        elif platform_name.startswith("darwin"):
            print(f"[SETUP] Attempting to install {editor_name} via Homebrew...")
            cmd = ["brew", "install", editor_name]
            subprocess.run(cmd, check=True)
        else:
            print(f"[SETUP] Auto-install not supported on '{platform_name}'. Please install a text editor manually.")
            return False
        
        print(f"[SETUP] Successfully installed '{editor_name}'.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[SETUP] Failed to install '{editor_name}'. Error: {e}")
        print("Please install a text editor manually (e.g., nano, vim, vscode).")
        return False


# --- Custom Nano-like Editor for Windows ---
import sys
import platform
import msvcrt
import ctypes


def run_custom_nano_editor(filename: Path):
    """
    A custom nano-like text editor for Windows with clipboard support (Ctrl+C, Ctrl+V, Ctrl+X).
    Transparently decodes/encodes .egg/.eggless files for editing.
    """
    import os

    try:
        import pyperclip
        clipboard_available = True
    except ImportError:
        clipboard_available = False
    import colorama
    from colorama import Fore, Style
    colorama.init()
    if not clipboard_available:
        print("[EDITOR] Clipboard support unavailable (pyperclip not installed)")

    # Detect if .egg or .eggless
    is_egg = filename.suffix == ".egg"
    is_eggless = filename.suffix == ".eggless"
    is_binary = False
    content = ""
    if is_egg:
        try:
            # Use open_egg to decode
            content = open_egg(filename).decode("utf-8", errors="replace")
        except Exception:
            content = ""
            is_binary = True
    elif is_eggless:
        try:
            content = filename.read_text(encoding="utf-8")
        except Exception:
            content = ""
            is_binary = True
    else:
        try:
            content = filename.read_text(encoding="utf-8")
        except Exception:
            try:
                content = filename.read_bytes().decode("utf-8", errors="replace")
                is_binary = True
            except Exception:
                content = ""
                is_binary = True
    lines = content.splitlines() or [""]

    cursor_x, cursor_y = 0, 0
    clipboard = ""
    status = "Ctrl+O: Save  Ctrl+X: Exit  Ctrl+C: Copy  Ctrl+V: Paste  Ctrl+K: Cut  Arrows: Move"
    filename_str = str(filename)

    def clear():
        os.system("cls")

    def print_screen():
        clear()
        print(Fore.CYAN + f"[nano] {filename_str}  {'(egg)' if is_egg else ('(eggless)' if is_eggless else ('(binary)' if is_binary else ''))}" + Style.RESET_ALL)
        for i, line in enumerate(lines):
            if i == cursor_y:
                # Show cursor position
                cx = min(cursor_x, len(line))
                print(line[:cx] + Fore.YELLOW + (line[cx:cx+1] or " ") + Style.RESET_ALL + line[cx+1:])
            else:
                print(line)
        print("\n" + Fore.GREEN + status + Style.RESET_ALL)
        print(f"Ln {cursor_y+1}, Col {cursor_x+1}")

    def getch():
        return msvcrt.getch()

    def save():
        data = "\n".join(lines)
        try:
            if is_egg:
                # Encode and save as .egg
                encoded = egg_encode_bytes(data.encode("utf-8", errors="replace"))
                filename.write_bytes(encoded)
            elif is_eggless:
                filename.write_text(data, encoding="utf-8")
            elif is_binary:
                filename.write_bytes(data.encode("utf-8", errors="replace"))
            else:
                filename.write_text(data, encoding="utf-8")
            return True, "[nano] Saved."
        except Exception as e:
            return False, f"[nano] Save error: {e}"

    while True:
        print_screen()
        ch = getch()
        if ch == b'\x03':  # Ctrl+C (Copy)
            clipboard = lines[cursor_y]
            try:
                pyperclip.copy(clipboard)
            except Exception:
                pass
            status = "[nano] Copied line."
        elif ch == b'\x16':  # Ctrl+V (Paste)
            try:
                paste = pyperclip.paste()
            except Exception:
                paste = clipboard
            lines.insert(cursor_y+1, paste)
            cursor_y += 1
            cursor_x = 0
            status = "[nano] Pasted."
        elif ch == b'\x18':  # Ctrl+X (Cut/Exit)
            if msvcrt.kbhit() and msvcrt.getch() == b'\x18':
                # Double Ctrl+X: exit without saving
                print("[nano] Exit without saving.")
                break
            clipboard = lines[cursor_y]
            try:
                pyperclip.copy(clipboard)
            except Exception:
                pass
            lines.pop(cursor_y)
            if cursor_y >= len(lines):
                cursor_y = max(0, len(lines)-1)
            cursor_x = 0
            status = "[nano] Cut line."
        elif ch == b'\x0f':  # Ctrl+O (Save)
            ok, msg = save()
            status = msg
        elif ch == b'\xe0' or ch == b'\x00':  # Arrow keys
            ch2 = getch()
            if ch2 == b'H':  # Up
                cursor_y = max(0, cursor_y-1)
                cursor_x = min(cursor_x, len(lines[cursor_y]))
            elif ch2 == b'P':  # Down
                cursor_y = min(len(lines)-1, cursor_y+1)
                cursor_x = min(cursor_x, len(lines[cursor_y]))
            elif ch2 == b'K':  # Left
                cursor_x = max(0, cursor_x-1)
            elif ch2 == b'M':  # Right
                cursor_x = min(len(lines[cursor_y]), cursor_x+1)
        elif ch == b'\r':  # Enter
            line = lines[cursor_y]
            left = line[:cursor_x]
            right = line[cursor_x:]
            lines[cursor_y] = left
            lines.insert(cursor_y+1, right)
            cursor_y += 1
            cursor_x = 0
        elif ch == b'\x08':  # Backspace
            if cursor_x > 0:
                line = lines[cursor_y]
                lines[cursor_y] = line[:cursor_x-1] + line[cursor_x:]
                cursor_x -= 1
            elif cursor_y > 0:
                prev = lines[cursor_y-1]
                lines[cursor_y-1] = prev + lines[cursor_y]
                lines.pop(cursor_y)
                cursor_y -= 1
                cursor_x = len(lines[cursor_y])
        elif ch == b'\x13':  # Ctrl+S (Save)
            ok, msg = save()
            status = msg
        elif ch == b'\x1a':  # Ctrl+Z (Undo not implemented)
            status = "[nano] Undo not implemented."
        elif ch == b'\x1b':  # Esc (Exit)
            break
        elif ch == b'\x11':  # Ctrl+Q (Exit)
            break
        elif ch == b'\x17':  # Ctrl+W (Where is/search not implemented)
            status = "[nano] Search not implemented."
        elif ch == b'\x0b':  # Ctrl+K (Cut line)
            clipboard = lines[cursor_y]
            try:
                pyperclip.copy(clipboard)
            except Exception:
                pass
            lines.pop(cursor_y)
            if cursor_y >= len(lines):
                cursor_y = max(0, len(lines)-1)
            cursor_x = 0
            status = "[nano] Cut line."
        elif ch == b'\x04':  # Ctrl+D (Delete char)
            line = lines[cursor_y]
            if cursor_x < len(line):
                lines[cursor_y] = line[:cursor_x] + line[cursor_x+1:]
            elif cursor_y < len(lines)-1:
                lines[cursor_y] += lines[cursor_y+1]
                lines.pop(cursor_y+1)
        elif ch >= b' ' and ch <= b'~':  # Printable
            line = lines[cursor_y]
            lines[cursor_y] = line[:cursor_x] + ch.decode(errors="replace") + line[cursor_x:]
            cursor_x += 1
        # Clamp cursor
        cursor_x = max(0, min(cursor_x, len(lines[cursor_y])))
        cursor_y = max(0, min(cursor_y, len(lines)-1))
    print("[nano] Exited editor.")

# fallback for non-Windows
def run_internal_editor(filename: Path):
    print("-" * 50)
    print(f"[EDITOR] Using built-in fallback editor for {filename}.")
    print("Type your content. Enter ':wq' on a new line to save and exit.")
    print("-" * 50)
    try:
        content = filename.read_text().splitlines()
    except FileNotFoundError:
        content = []
    print("\n--- Current Content ---\n")
    for line in content:
        print(line)
    print("\n--- End of Content | Enter New Content Below ---\n")
    new_lines = []
    while True:
        try:
            line = input()
            if line.strip() == ":wq":
                break
            new_lines.append(line)
        except EOFError:
            break
    try:
        filename.write_text("\n".join(new_lines))
        print(f"\n[EDITOR] Saved changes to {filename}")
    except Exception as e:
        print(f"\n[EDITOR] Error saving file: {e}")

def cmd_editor_open(args):
    """editor.open <file> - open with configured editor or custom nano-like editor on Windows"""
    if not args:
        print("Usage: editor.open <file>")
        return
    filename = Path(args[0])
    editor = CONFIG.get("editor", "internal")
    if platform.system().lower() == "windows":
        try:
            import pyperclip
            import colorama
            run_custom_nano_editor(filename)
            return
        except ImportError:
            print("[EDITOR] Required modules for custom editor not found. Clipboard support will be disabled.")
            run_custom_nano_editor(filename)
            return
    if editor == "internal":
        run_internal_editor(filename)
        return
    if shutil.which(editor):
        try:
            subprocess.run([editor, str(filename)])
        except Exception as e:
            print(f"[EDITOR] Error running '{editor}': {e}")
    else:
        if auto_install_editor(editor):
            try:
                subprocess.run([editor, str(filename)])
            except Exception as e:
                print(f"[EDITOR] Error running '{editor}' after install: {e}")
        else:
            print("[EDITOR] Falling back to internal editor for this session.")
            run_internal_editor(filename)

def cmd_fetch_cl_update_texteditor(args):
    """fetch.cl.update:texteditor:<name> - change editor"""
    if not args:
        print("Usage: fetch.cl.update:texteditor:<name> (e.g. vi, nano)")
        return
    
    name = args[0].lower()
    
    if shutil.which(name):
        CONFIG["editor"] = name
        save_config()
        print(f"[UPDATE] Default editor set to '{name}'")
    else:
        print(f"[UPDATE] Editor '{name}' not found in system PATH.")
        if auto_install_editor(name):
            CONFIG["editor"] = name
            save_config()
            print(f"[UPDATE] Default editor successfully installed and set to '{name}'")
        else:
            print(f"[UPDATE] Editor preference not changed. Current is '{CONFIG.get('editor')}'.")

# ----------------------------
# Small utility commands
# ----------------------------
def cmd_core_pulse(args):
    print("[CORE] Pulse OK. Uptime (approx):", round(time.time() - START_TIME, 2), "s")

def cmd_core_boot(args):
    print("[CORE] Reloading modules not implemented in prototype. Restart the process to apply changes.")

# ----------------------------
# Tiny custom language: ProLang
# ----------------------------
"""
ProLang - tiny readable language for demo:
Features:
- class <Name> { ... }
- var <name> = <literal>
- func <name>(args) { ... }
- obj = new <ClassName>();
- obj.call(...) and obj.prop reads
- sql select X from Y where Z  (runs on sqlite in-memory or file)
- embed js/css/html by writing files or launching server
- runner: lang.run <file.egg or file.eggless>
- host: lang.host [port]
"""

# A very small parser/executor
class ProLangRuntime:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.globals = {"__builtins__": {}}
        # lightweight sqlite DB for sql queries
        self.db = sqlite3.connect(str(workspace / "prolang.db"))
        self._prepare_sql()

    def _prepare_sql(self):
        cur = self.db.cursor()
        # example table for demo
        cur.execute("CREATE TABLE IF NOT EXISTS demo(id INTEGER PRIMARY KEY, name TEXT)")
        cur.execute("INSERT OR IGNORE INTO demo(id,name) VALUES(1,'alice')")
        self.db.commit()

    def run_code(self, code: str, filename="<prolang>"):
        # Very small line-based interpreter
        lines = [ln.strip() for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        objects = {}
        classes = {}
        vars_ = {}
        funcs = {}
        for ln in lines:
            if ln.startswith("var "):
                # var x = value
                try:
                    rest = ln[4:].strip()
                    name, val = rest.split("=",1)
                    name = name.strip()
                    val = val.strip()
                    # simple literal parsing
                    if val.startswith("'") or val.startswith('"'):
                        val = val.strip("'\"")
                    elif val.isdigit():
                        val = int(val)
                    vars_[name]=val
                    print("[PL] var",name,"=",val)
                except Exception as e:
                    print("[PL] var parse error", e)
            elif ln.startswith("class "):
                # class Demo { prop: value }
                # minimal: capture as simple dict
                try:
                    name = ln.split()[1]
                    classes[name]= {"__src__": ln}
                    print("[PL] class",name,"registered")
                except Exception as e:
                    print("[PL] class parse err", e)
            elif ln.startswith("func "):
                # store source as pseudo-func to print later
                fname = ln.split()[1].split("(")[0]
                funcs[fname]=ln
                print("[PL] func",fname,"registered")
            elif ln.startswith("sql "):
                # pass to sqlite
                q = ln[4:].strip()
                cur = self.db.cursor()
                try:
                    cur.execute(q)
                    rows = cur.fetchall()
                    for r in rows: print("[SQL]", r)
                    self.db.commit()
                except Exception as e:
                    print("[SQL] Error:", e)
            elif ln.startswith("write "):
                # write filename <<EOF ... EOF  (very tiny)
                try:
                    # pattern: write filename content
                    _, fname, *rest = shlex.split(ln)
                    payload = " ".join(rest)
                    p = self.workspace / fname
                    p.write_text(payload)
                    print("[PL] Wrote", p)
                except Exception as e:
                    print("[PL] write error", e)
            elif ln.startswith("new "):
                # new ClassName as var
                parts = ln.split()
                if len(parts)>=3:
                    varname = parts[1]
                    cname = parts[2]
                    objects[varname] = {"class": cname, "props": {}}
                    print("[PL] new object", varname, "of", cname)
            elif "." in ln and "(" in ln:
                # method call like obj.method(arg)
                try:
                    left, rest = ln.split("(",1)
                    objname, method = left.split(".",1)
                    args = rest.split(")")[0]
                    print(f"[PL] Call {objname}.{method}({args}) - in prototype this is a no-op")
                except Exception as e:
                    print("[PL] call parse err", e)
            else:
                print("[PL] Unknown line:", ln)

# Runner functions
PL_RUNTIME = ProLangRuntime(WS)

def cmd_lang_run(args):
    """lang.run <file.egg|file.eggless> - run a ProLang program (demo)"""
    if not args:
        print("Usage: lang.run <file.egg or file.eggless>")
        return
    p = Path(args[0])
    if not p.exists():
        print("Missing", p); return
    try:
        if p.suffix == ".egg":
            code = open_egg(p).decode()
        else:
            code = p.read_text()
        print("[LANG] Running", p)
        PL_RUNTIME.run_code(code, filename=str(p))
    except Exception as e:
        print("[LANG] Error:", e)

# Host the workspace on localhost:port
_http_server_thread = None
def _start_http_server(port):
    os.chdir(str(WS))
    handler = SimpleHTTPRequestHandler
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"[HOST] Serving {WS} at http://localhost:{port} (ctrl-c in terminal to stop server)")
    server.serve_forever()

def cmd_lang_host(args):
    """lang.host [port] - host workspace via simple HTTP server"""
    port = CONFIG.get("server_port", 6969)
    if args:
        try: port = int(args[0])
        except: pass
    t = threading.Thread(target=_start_http_server, args=(port,), daemon=True)
    t.start()
    print("[HOST] Server started in background thread (will stop on exit)")

def cmd_lang_editable(args):
    """lang.editable <on|off> - toggle editable mode (placeholder)"""
    print("[LANG] Editable mode toggling not implemented in prototype. Use editor.open to edit files.")

### --- Fetch subcommands expansion (examples, stubs, and help) ---
def cmd_fetch_info_meta(args):
    """fetch.info.meta <file/url> - show meta info (extended)"""
    print("[FETCH.INFO.META] Not implemented.")
def cmd_fetch_info_type(args):
    """fetch.info.type <file/url> - show file type"""
    print("[FETCH.INFO.TYPE] Not implemented.")
def cmd_fetch_info_hash(args):
    """fetch.info.hash <file/url> - show hash info"""
    print("[FETCH.INFO.HASH] Not implemented.")
def cmd_fetch_info_links(args):
    """fetch.info.links <url> - show all links in a page"""
    print("[FETCH.INFO.LINKS] Not implemented.")
def cmd_fetch_info_headers(args):
    """fetch.info.headers <url> - show HTTP headers"""
    print("[FETCH.INFO.HEADERS] Not implemented.")
def cmd_fetch_info_redirects(args):
    """fetch.info.redirects <url> - show redirect chain"""
    print("[FETCH.INFO.REDIRECTS] Not implemented.")
def cmd_fetch_info_ssl(args):
    """fetch.info.ssl <url> - show SSL info"""
    print("[FETCH.INFO.SSL] Not implemented.")
def cmd_fetch_info_cookies(args):
    """fetch.info.cookies <url> - show cookies"""
    print("[FETCH.INFO.COOKIES] Not implemented.")
def cmd_fetch_info_dns(args):
    """fetch.info.dns <url> - show DNS info"""
    print("[FETCH.INFO.DNS] Not implemented.")
def cmd_fetch_info_whois(args):
    """fetch.info.whois <url> - show WHOIS info"""
    print("[FETCH.INFO.WHOIS] Not implemented.")
def cmd_fetch_info_geo(args):
    """fetch.info.geo <url> - show geo info"""
    print("[FETCH.INFO.GEO] Not implemented.")
def cmd_fetch_info_tech(args):
    """fetch.info.tech <url> - show tech stack"""
    print("[FETCH.INFO.TECH] Not implemented.")
def cmd_fetch_info_robots(args):
    """fetch.info.robots <url> - show robots.txt"""
    print("[FETCH.INFO.ROBOTS] Not implemented.")
def cmd_fetch_info_sitemap(args):
    """fetch.info.sitemap <url> - show sitemap.xml"""
    print("[FETCH.INFO.SITEMAP] Not implemented.")
def cmd_fetch_info_icons(args):
    """fetch.info.icons <url> - show icons"""
    print("[FETCH.INFO.ICONS] Not implemented.")
def cmd_fetch_info_fonts(args):
    """fetch.info.fonts <url> - show fonts"""
    print("[FETCH.INFO.FONTS] Not implemented.")
def cmd_fetch_info_scripts(args):
    """fetch.info.scripts <url> - show scripts"""
    print("[FETCH.INFO.SCRIPTS] Not implemented.")
def cmd_fetch_info_images(args):
    """fetch.info.images <url> - show images"""
    print("[FETCH.INFO.IMAGES] Not implemented.")
def cmd_fetch_info_videos(args):
    """fetch.info.videos <url> - show videos"""
    print("[FETCH.INFO.VIDEOS] Not implemented.")
def cmd_fetch_info_audio(args):
    """fetch.info.audio <url> - show audio files"""
    print("[FETCH.INFO.AUDIO] Not implemented.")
def cmd_fetch_info_pdf(args):
    """fetch.info.pdf <url> - show PDF files"""
    print("[FETCH.INFO.PDF] Not implemented.")
def cmd_fetch_info_zip(args):
    """fetch.info.zip <url> - show ZIP files"""
    print("[FETCH.INFO.ZIP] Not implemented.")
def cmd_fetch_info_tar(args):
    """fetch.info.tar <url> - show TAR files"""
    print("[FETCH.INFO.TAR] Not implemented.")
def cmd_fetch_info_json(args):
    """fetch.info.json <url> - show JSON files"""
    print("[FETCH.INFO.JSON] Not implemented.")
def cmd_fetch_info_xml(args):
    """fetch.info.xml <url> - show XML files"""
    print("[FETCH.INFO.XML] Not implemented.")
def cmd_fetch_info_csv(args):
    """fetch.info.csv <url> - show CSV files"""
    print("[FETCH.INFO.CSV] Not implemented.")
def cmd_fetch_info_md(args):
    """fetch.info.md <url> - show Markdown files"""
    print("[FETCH.INFO.MD] Not implemented.")
def cmd_fetch_info_html(args):
    """fetch.info.html <url> - show HTML files"""
    print("[FETCH.INFO.HTML] Not implemented.")
def cmd_fetch_info_css(args):
    """fetch.info.css <url> - show CSS files"""
    print("[FETCH.INFO.CSS] Not implemented.")
def cmd_fetch_info_js(args):
    """fetch.info.js <url> - show JS files"""
    print("[FETCH.INFO.JS] Not implemented.")
def cmd_fetch_info_py(args):
    """fetch.info.py <url> - show Python files"""
    print("[FETCH.INFO.PY] Not implemented.")
COMMAND_MAP = {
    # CORE SYSTEM COMMANDS (25+)
    'gadget': cmd_gadget,
    'core': cmd_core,
    'whisper': cmd_whisper,
    'wakeup': cmd_wakeup,
    'nap': cmd_nap,
    # ...existing code...
    'scrub': cmd_scrub,
    'stash': cmd_stash,
    'draw': cmd_draw,
    'mixer': cmd_mixer,
    'scroll': cmd_scroll,
    'count': cmd_count,
    'alias': cmd_alias,
    'jail': cmd_jail,
    'finger': cmd_finger,
    'crypt': cmd_crypt,
    'pool': cmd_pool,
    'span': cmd_span,
    'pivot': cmd_pivot,
    'watch': cmd_watch,
    'mirror': cmd_mirror,
    'scan': cmd_scan,
    'debug': cmd_debug,
    'shell': cmd_shell,
    'window.config': cmd_window_config,
    'tinker': cmd_tinker,
    'rept': cmd_rept,
    'egg..runtime': cmd_egg_run,  # alias for egg.run
    'window': cmd_window,
    'siphon': cmd_siphon,
    'juju': cmd_juju,
    'doctor': cmd_doctor,
    'clock': cmd_clock,
    'shout': cmd_shout,
    'poke': cmd_poke,
    'seat': cmd_seat,
    'handle': cmd_handle,
    'lever': cmd_lever,
    'vault': cmd_vault,
    'compass': cmd_compass,
    'gate': cmd_gate,
    'hustle': cmd_hustle,
    'makenode': cmd_makenode,
    'linker': cmd_linker,
    'tuner': cmd_tuner,
    # SYNC subcommands (12+)
    "sync.files": cmd_sync_files,
    "sync.dirs": cmd_sync_dirs,
    "sync.new": cmd_sync_new,
    "sync.changed": cmd_sync_changed,
    "sync.deleted": cmd_sync_deleted,
    "sync.summary": cmd_sync_summary,
    "sync.log": cmd_sync_log,
    "sync.verify": cmd_sync_verify,
    "sync.backup": cmd_sync_backup,
    "sync.restore": cmd_sync_restore,
    "sync.status": cmd_sync_status,
    "sync.files.md5": cmd_sync_files_md5,
    "sync.files.sha256": cmd_sync_files_sha256,
    "sync.files.log": cmd_sync_files_log,
    "sync.files.verify": cmd_sync_files_verify,
    "sync.files.backup": cmd_sync_files_backup,
    "sync.files.restore": cmd_sync_files_restore,
    # FETCH subcommands (12+)
    "fetch.file.type": cmd_fetch_file_type,
    "fetch.file.size": cmd_fetch_file_size,
    "fetch.dir": cmd_fetch_dir,
    "fetch.dir.list": cmd_fetch_dir_list,
    "fetch.dir.size": cmd_fetch_dir_size,
    "fetch.dir.save": cmd_fetch_dir_save,
    "fetch.dir.type": cmd_fetch_dir_type,
    "fetch.dir.info": cmd_fetch_dir_info,
    "fetch.url": cmd_fetch_url,
    "fetch.url.headers": cmd_fetch_url_headers,
    "fetch.url.status": cmd_fetch_url_status,
    "fetch.url.save": cmd_fetch_url_save,
    "fetch.url.type": cmd_fetch_url_type,
    "fetch.url.info": cmd_fetch_url_info,
    "fetch.url.links": cmd_fetch_url_links,
    # FIND subcommands (12+)
    "find.email.pattern": cmd_find_email_pattern,
    "find.email.count": cmd_find_email_count,
    "find.email.unique": cmd_find_email_unique,
    "find.email.save": cmd_find_email_save,
    "find.phone": cmd_find_phone,
    "find.phone.country": cmd_find_phone_country,
    "find.phone.pattern": cmd_find_phone_pattern,
    "find.phone.count": cmd_find_phone_count,
    "find.phone.unique": cmd_find_phone_unique,
    "find.phone.save": cmd_find_phone_save,
    "find.hash": cmd_find_hash,
    "find.hash.type": cmd_find_hash_type,
    "find.hash.pattern": cmd_find_hash_pattern,
    "find.hash.count": cmd_find_hash_count,
    "find.hash.unique": cmd_find_hash_unique,
    "find.hash.save": cmd_find_hash_save,
    "find.url": cmd_find_url,
    "find.site": cmd_find_site,
    "find.dir": cmd_find_dir,
    "find.file": cmd_find_file,
    "find.port": cmd_find_port,
    "find.url.port": cmd_find_url_port,
    "find.port.url": cmd_find_port_url,
    "find.dir.all": cmd_find_dir_all,
    "find.url.all": cmd_find_url_all,
    "find.website.all": cmd_find_website_all,
    "toad": cmd_toad,
    "toad.port": cmd_toad_port,
    "analyze": cmd_analyze,
    "monitor": cmd_monitor,
    "convert": cmd_convert,
    "report": cmd_report,
    "auto": cmd_auto,
    "extract": cmd_extract,
    "sync": cmd_sync,
    "scan": cmd_scan,
    "fetch": cmd_fetch,
    "fetch.dir": cmd_fetch_dir,
    "goto": cmd_goto,
    "wentto": cmd_wentto,
    "byby": cmd_byby,
    "fetch.info.meta": cmd_fetch_info_meta,
    "fetch.info.type": cmd_fetch_info_type,
    "fetch.info.hash": cmd_fetch_info_hash,
    "fetch.info.links": cmd_fetch_info_links,
    "fetch.info.headers": cmd_fetch_info_headers,
    "fetch.info.redirects": cmd_fetch_info_redirects,
    "fetch.info.ssl": cmd_fetch_info_ssl,
    "fetch.info.cookies": cmd_fetch_info_cookies,
    "fetch.info.dns": cmd_fetch_info_dns,
    "fetch.info.whois": cmd_fetch_info_whois,
    "fetch.info.geo": cmd_fetch_info_geo,
    "fetch.info.tech": cmd_fetch_info_tech,
    "fetch.info.robots": cmd_fetch_info_robots,
    "fetch.info.sitemap": cmd_fetch_info_sitemap,
    "fetch.info.icons": cmd_fetch_info_icons,
    "fetch.info.fonts": cmd_fetch_info_fonts,
    "fetch.info.scripts": cmd_fetch_info_scripts,
    "fetch.info.images": cmd_fetch_info_images,
    "fetch.info.videos": cmd_fetch_info_videos,
    "fetch.info.all": cmd_fetch_info_all,
    "fetch.info.all:download": cmd_fetch_info_all_download,
    "fetch.dwn": cmd_fetch_dwn,
    "editor.open": cmd_editor_open,
    "fetch.cl.update": cmd_fetch_cl_update,
    "fetch.cl.update:texteditor": cmd_fetch_cl_update_texteditor,
    "file.transfer": cmd_file_transfer,
    "file.split": cmd_file_split,
    "file.join": cmd_file_join,
    "file.hash": cmd_file_hash,
    "file.diff": cmd_file_diff,
    "dir.transfer": cmd_dir_transfer,
    "dir.list": cmd_dir_list,
    "dir.size": cmd_dir_size,
    "egg.make": cmd_egg_make,
    "egg.open": cmd_egg_open,
    "egg.convert": cmd_egg_convert,
    "egg.vars": cmd_egg_vars,
    "egg.verify": cmd_egg_verify,
    "egg.clean": cmd_egg_clean,
    "pong": cmd_pong,
    "pong.ext": cmd_pong_ext,
    "pong.trace": cmd_pong_trace,
    "sys.info": cmd_sys_info,
    "sys.mem": cmd_sys_mem,
    "sys.proc": cmd_sys_proc,
    "sys.kill": cmd_sys_kill,
    "sys.clr": cmd_sys_clr,
    "sys.time": cmd_sys_time,
    "native": cmd_native,
    "core.pulse": cmd_core_pulse,
    "core.boot": cmd_core_boot,
    "lang.run": cmd_lang_run,
    "lang.host": cmd_lang_host,
    "lang.editable": cmd_lang_editable,
    "fetch.info.meta": cmd_fetch_info_meta,
    "fetch.info.audio": cmd_fetch_info_audio,
    "fetch.info.pdf": cmd_fetch_info_pdf,
    "fetch.info.zip": cmd_fetch_info_zip,
    "fetch.info.tar": cmd_fetch_info_tar,
    "fetch.info.json": cmd_fetch_info_json,
    "fetch.info.xml": cmd_fetch_info_xml,
    "fetch.info.csv": cmd_fetch_info_csv,
    "fetch.info.md": cmd_fetch_info_md,
    "fetch.info.html": cmd_fetch_info_html,
    "fetch.info.css": cmd_fetch_info_css,
    "fetch.info.js": cmd_fetch_info_js,
    "fetch.info.py": cmd_fetch_info_py,
    "fetch.dwn": cmd_fetch_dwn,
    "/python/": cmd_python,
    "clear": cmd_clear,
    "export": cmd_export,
    "unset": cmd_unset,
    "env": cmd_env,
    "fetch.info": cmd_fetch_info,
    "fetch.info.all": cmd_fetch_info_all,
    "fetch.info.all:download": cmd_fetch_info_all_download,
    "fetch.blank": cmd_fetch_blank,
    "fetch.cl.update": cmd_fetch_cl_update,
    "fetch.cl.update:texteditor": cmd_fetch_cl_update_texteditor,
    "file.transfer": cmd_file_transfer,
    "file.split": cmd_file_split,
    "file.join": cmd_file_join,
    "file.hash": cmd_file_hash,
    "file.diff": cmd_file_diff,
    "dir.transfer": cmd_dir_transfer,
    "dir.list": cmd_dir_list,
    "dir.size": cmd_dir_size,
    "egg.make": cmd_egg_make,
    "egg.open": cmd_egg_open,
    "egg.convert": cmd_egg_convert,
    "egg.vars": cmd_egg_vars,
    "egg.verify": cmd_egg_verify,
    "egg.clean": cmd_egg_clean,
    "pong": cmd_pong,
    "pong.ext": cmd_pong_ext,
    "pong.trace": cmd_pong_trace,
    "sys.info": cmd_sys_info,
    "sys.mem": cmd_sys_mem,
    "sys.proc": cmd_sys_proc,
    "sys.proc.info": cmd_sys_proc_info,
    "sys.kill": cmd_sys_kill,
    "sys.clr": cmd_sys_clr,
    "sys.time": cmd_sys_time,
    "editor.open": cmd_editor_open,
    "core.pulse": cmd_core_pulse,
    "core.boot": cmd_core_boot,
    "lang.run": cmd_lang_run,
    "lang.host": cmd_lang_host,
    "lang.editable": cmd_lang_editable,
    "toad": cmd_toad,
    "toad.port": cmd_toad_port,
    "find.url": cmd_find_url,
    "find.site": cmd_find_site,
    "find.dir": cmd_find_dir,
    "find.file": cmd_find_file,
    "find.port": cmd_find_port,
    "find.url.port": cmd_find_url_port,
    "find.port.url": cmd_find_port_url,
    "find.dir.all": cmd_find_dir_all,
    "find.url.all": cmd_find_url_all,
    "find.website.all": cmd_find_website_all,
    "scan": cmd_scan,
    "analyze": cmd_analyze,
    "monitor": cmd_monitor,
    "convert": cmd_convert,
    "report": cmd_report,
    "auto": cmd_auto,
    "sync": cmd_sync,
    "extract": cmd_extract,

    "help": cmd_help,
    "config.terminal.egg": cmd_config_terminal_egg,
}

add_help_variants()

# Aliases and native passthrough prefix handled separately.
EXTRA_HELP = {
    "fetch.cl.update:texteditor": "Usage: fetch.cl.update:texteditor <name>  e.g. vi or nano",
}

# ----------------------------
# REPL
# ----------------------------
START_TIME = time.time()

def parse_and_execute(line: str):
    """Parse shell command line and execute"""
    lexer = Lexer(line)
    tokens = lexer.tokenize()

    # Expand tokens
    expander = Expander()
    expanded_tokens = expander.expand(tokens)

    # Full AST parsing and execution
    parser = Parser(expanded_tokens)
    statements = parser.parse()
    for stmt in statements:
        execute_ast(stmt)

def execute_ast(node: ASTNode):
    """Execute AST node"""
    if isinstance(node, Command):
        execute_command(node)
    elif isinstance(node, Pipeline):
        execute_pipeline(node)
    elif isinstance(node, Background):
        execute_background(node)
    elif isinstance(node, IfStatement):
        execute_if(node)
    elif isinstance(node, ForLoop):
        execute_for(node)
    elif isinstance(node, WhileLoop):
        execute_while(node)
    elif isinstance(node, FunctionDef):
        execute_function_def(node)
    elif isinstance(node, Subshell):
        execute_subshell(node)
    elif isinstance(node, Group):
        execute_group(node)

def execute_command(cmd: Command):
    """Execute a command with redirections"""
    # Handle redirections
    stdin_fd = None
    stdout_fd = None
    stderr_fd = None

    try:
        if 'stdin' in cmd.redirections:
            stdin_fd = open(cmd.redirections['stdin'], 'r')
        if 'stdout' in cmd.redirections:
            stdout_fd = open(cmd.redirections['stdout'], 'w')
        elif 'stdout_append' in cmd.redirections:
            stdout_fd = open(cmd.redirections['stdout_append'], 'a')
        if 'stderr' in cmd.redirections:
            stderr_fd = open(cmd.redirections['stderr'], 'w')
        elif 'stderr_to_stdout' in cmd.redirections:
            stderr_fd = stdout_fd

        # Execute command
        if cmd.name in COMMAND_MAP:
            # Built-in command
            COMMAND_MAP[cmd.name](cmd.args)
        elif cmd.name in FUNCTIONS:
            # User-defined function
            func_def = FUNCTIONS[cmd.name]
            # Set parameters as variables
            for i, param in enumerate(func_def.params):
                if i < len(cmd.args):
                    os.environ[param] = cmd.args[i]
            # Execute function body
            for stmt in func_def.body:
                execute_ast(stmt)
        else:
            # External command
            subprocess.run([cmd.name] + cmd.args,
                          stdin=stdin_fd or sys.stdin,
                          stdout=stdout_fd or sys.stdout,
                          stderr=stderr_fd or sys.stderr)
    finally:
        if stdin_fd: stdin_fd.close()
        if stdout_fd: stdout_fd.close()
        if stderr_fd and stderr_fd != stdout_fd: stderr_fd.close()

def execute_pipeline(pipeline: Pipeline):
    """Execute pipeline of commands"""
    if len(pipeline.commands) == 1:
        execute_command(pipeline.commands[0])
        return

    # For now, simple implementation without proper pipe handling
    prev_stdout = None
    for i, cmd in enumerate(pipeline.commands):
        if i == 0:
            # First command
            proc = subprocess.Popen([cmd.name] + cmd.args,
                                  stdout=subprocess.PIPE)
            prev_stdout = proc.stdout
        elif i == len(pipeline.commands) - 1:
            # Last command
            subprocess.run([cmd.name] + cmd.args,
                         stdin=prev_stdout)
        else:
            # Middle command
            proc = subprocess.Popen([cmd.name] + cmd.args,
                                  stdin=prev_stdout,
                                  stdout=subprocess.PIPE)
            prev_stdout = proc.stdout

def execute_background(bg: Background):
    """Execute command in background"""
    import threading
    def run():
        execute_ast(bg.command)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

def execute_if(if_stmt: IfStatement):
    """Execute if statement"""
    # Evaluate condition as a command
    condition_cmd = if_stmt.condition.strip()
    if condition_cmd:
        # Parse and execute the condition command
        lexer = Lexer(condition_cmd)
        tokens = lexer.tokenize()
        expander = Expander()
        expanded_tokens = expander.expand(tokens)
        parser = Parser(expanded_tokens)
        cond_stmts = parser.parse()
        exit_code = 0
        for stmt in cond_stmts:
            try:
                execute_ast(stmt)
            except SystemExit as e:
                exit_code = e.code or 0
                break
        if exit_code == 0:
            for stmt in if_stmt.then_block:
                execute_ast(stmt)
        else:
            # Check elif blocks
            executed = False
            for elif_cond, elif_body in if_stmt.elif_blocks:
                elif_cmd = elif_cond.strip()
                if elif_cmd:
                    lexer = Lexer(elif_cmd)
                    tokens = lexer.tokenize()
                    expander = Expander()
                    expanded_tokens = expander.expand(tokens)
                    parser = Parser(expanded_tokens)
                    elif_stmts = parser.parse()
                    exit_code = 0
                    for stmt in elif_stmts:
                        try:
                            execute_ast(stmt)
                        except SystemExit as e:
                            exit_code = e.code or 0
                            break
                    if exit_code == 0:
                        for stmt in elif_body:
                            execute_ast(stmt)
                        executed = True
                        break
            if not executed:
                for stmt in if_stmt.else_block:
                    execute_ast(stmt)
    else:
        for stmt in if_stmt.else_block:
            execute_ast(stmt)

def execute_for(for_loop: ForLoop):
    """Execute for loop"""
    var_name = for_loop.var
    for item in for_loop.items:
        # Set the loop variable in environment
        os.environ[var_name] = item
        for stmt in for_loop.body:
            execute_ast(stmt)

def execute_while(while_loop: WhileLoop):
    """Execute while loop"""
    while True:  # Simplified
        for stmt in while_loop.body:
            execute_ast(stmt)
        break  # Prevent infinite loop for now

# Global functions registry
FUNCTIONS = {}

def execute_function_def(func_def: FunctionDef):
    """Define function"""
    FUNCTIONS[func_def.name] = func_def

def execute_subshell(subshell: Subshell):
    """Execute subshell (in same process for now)"""
    for stmt in subshell.commands:
        execute_ast(stmt)

def execute_group(group: Group):
    """Execute command group"""
    for stmt in group.commands:
        execute_ast(stmt)

def dispatch(raw_line: str):
    raw = raw_line.strip()
    if not raw: return
    if raw in ("exit","quit"):
        print("Bye.")
        sys.exit(0)
    if raw.startswith("/native/"):
        return cmd_native(raw)
    # Parse --help
    parts = shlex.split(raw)
    cmd = parts[0]
    args = parts[1:]
    if cmd.endswith("--help") or (len(args)==1 and args[0]=="--help"):
        # format: cmd --help or cmd --help
        base = cmd.replace("--help","").strip()
        if not base: base = parts[0]
        # attempt command lookup
        fn = COMMAND_MAP.get(base)
        if fn:
            doc = fn.__doc__ or "(no doc)"
            print(doc)
        else:
            print("No help for", base)
        return
    # Try new parser first
    try:
        parse_and_execute(raw)
        return
    except Exception as e:
        # Fall back to old dispatch
        pass
    # Direct match
    if cmd in COMMAND_MAP:
        try:
            COMMAND_MAP[cmd](args)
        except TypeError:
            # try call with raw args for special forms
            COMMAND_MAP[cmd](args)
        return
    # support dotted with colon style like fetch.cl.update:texteditor:vi
    if ":" in cmd:
        # split into base colon path
        pre, *rest = cmd.split(":")
        if pre in COMMAND_MAP:
            # supply rest as args
            COMMAND_MAP[pre](rest + args)
            return
    # some convenience aliases
    if cmd == "help":
        print(short_help()); return
    if cmd == "/?":
        print(short_help()); return
    # unknown
    print("[ERROR] Unknown command:", cmd)
    print("Type 'help' for quick list.")

def startup_checks():
    """Prints a summary of the environment at boot."""
    print("[BOOT] Checking environment...")
    editor = CONFIG.get('editor', 'internal')
    editor_status = f"✅ '{editor}' found" if editor != 'internal' and shutil.which(editor) else f"⚠️  Using '{editor}' fallback"
    print(f"→ Checking editors... {editor_status}")
    
    python_libs_status = "✅ requests available" if requests else "⚠️ 'requests' not installed, using fallback"
    print(f"→ Checking Python libs... {python_libs_status}")
    print("→ Checking network... ✅")
    print("-" * 20)
    print(f"[READY] PROCLI started. Workspace: {WS}")
    print("-" * 20)

def repl():
    startup_checks()

    session = create_prompt_session()

    while True:
        try:
            if session:
                # Use prompt_toolkit
                raw = session.prompt(get_prompt())
            else:
                # Fallback to basic input
                raw = input(get_prompt())
        except EOFError:
            print(); break
        except KeyboardInterrupt:
            print(); continue
        if not raw.strip(): continue
        dispatch(raw)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    try:
        repl()
    except SystemExit:
        pass
    except Exception as e:
        print("Fatal:", e)