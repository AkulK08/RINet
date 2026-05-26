"""Domain-specific scripting language for RINet / AlloGraph experiments.

The module implements a compact compiler-style stack for residue-interaction
network workflows:

* token definitions and a source-position-aware lexer;
* a recursive-descent parser that builds typed AST nodes;
* semantic validation with a small symbol table;
* runtime values for proteins, graph bundles, signals, spectra, and reports;
* an interpreter that connects scripts to the existing AlloGraph core package;
* execution hooks so future engines can override graph construction, spectral
  analysis, counterfactual analysis, diffusion, and export behavior;
* a formatter and a minimal REPL.

The goal is not to invent a general-purpose programming language. RINetScript is
intentionally declarative and experiment-shaped. A typical script looks like:

    LOAD "protein.pdb" AS protein
    GRAPH protein CUTOFF 8.0 AS G
    SEED G NODE 45 STRENGTH 1.0 AS signal
    DIFFUSE signal STEPS 100 DT 0.05 AS output
    SPECTRAL G MODES 20 AS spectrum
    COUNTERFACTUAL G REMOVE_NODE 88 AS cf
    EXPORT output TO "results.csv"

The interpreter is designed to be usable today with the lightweight package that
currently exists, while also exposing hook points for the larger spectral,
counterfactual, robustness, and orchestration modules that can be added later.
"""

from __future__ import annotations

import cmd
import csv
import json
import math
import os
import pathlib
import re
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .graphio.io import adjacency_from_csv, bundle_from_adjacency, bundle_from_npz, bundle_to_npz
from .graphio.pdb import pdb_to_rin
from .graphio.synth import synthetic_rin
from .graphio.types import GraphBundle
from .dynamics import make_model
from .inference.forward import ForwardResult, run_forward


# =============================================================================
# Source locations and diagnostics
# =============================================================================


@dataclass(frozen=True)
class SourceLocation:
    """A precise source position used in compiler-style errors."""

    line: int
    column: int
    index: int

    def as_dict(self) -> Dict[str, int]:
        return {"line": self.line, "column": self.column, "index": self.index}


@dataclass(frozen=True)
class SourceSpan:
    """A half-open source span from start to end."""

    start: SourceLocation
    end: SourceLocation

    def as_dict(self) -> Dict[str, Dict[str, int]]:
        return {"start": self.start.as_dict(), "end": self.end.as_dict()}


class RINetLanguageError(Exception):
    """Base class for all RINetScript errors."""

    def __init__(self, message: str, span: Optional[SourceSpan] = None, source: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.span = span
        self.source = source

    def render(self, source: Optional[str] = None) -> str:
        source_text = source if source is not None else self.source
        if self.span is None or not source_text:
            return self.message
        line_no = self.span.start.line
        col = self.span.start.column
        lines = source_text.splitlines()
        if 1 <= line_no <= len(lines):
            line_text = lines[line_no - 1]
            width = max(1, self.span.end.column - self.span.start.column)
            pointer = " " * max(0, col - 1) + "^" * width
            return f"{self.message}\n  at line {line_no}, column {col}\n  {line_text}\n  {pointer}"
        return f"{self.message}\n  at line {line_no}, column {col}"


class LexerError(RINetLanguageError):
    """Raised when source text cannot be tokenized."""


class ParserError(RINetLanguageError):
    """Raised when tokenized input does not match the grammar."""


class SemanticError(RINetLanguageError):
    """Raised when a syntactically valid script is semantically invalid."""


class RuntimeEvaluationError(RINetLanguageError):
    """Raised during execution of a valid script."""


# =============================================================================
# Token definitions and lexer
# =============================================================================


class TokenKind(Enum):
    EOF = auto()
    NEWLINE = auto()
    SEMICOLON = auto()
    COMMA = auto()
    COLON = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LPAREN = auto()
    RPAREN = auto()
    IDENT = auto()
    STRING = auto()
    INT = auto()
    FLOAT = auto()
    BOOLEAN = auto()

    LOAD = auto()
    GRAPH = auto()
    SEED = auto()
    DIFFUSE = auto()
    SPECTRAL = auto()
    COUNTERFACTUAL = auto()
    EXPORT = auto()
    LET = auto()
    SHOW = auto()
    DESCRIBE = auto()
    HELP = auto()

    AS = auto()
    TO = auto()
    FROM = auto()
    WITH = auto()
    USING = auto()
    WHERE = auto()
    AND = auto()
    OR = auto()

    CUTOFF = auto()
    CHAIN = auto()
    WEIGHT = auto()
    MODE = auto()
    MODEL = auto()
    CONTACT = auto()
    DEMO = auto()
    NPZ = auto()
    CSV = auto()
    PDB = auto()

    NODE = auto()
    NODES = auto()
    EDGE = auto()
    EDGES = auto()
    PATH = auto()
    STRENGTH = auto()
    VALUES = auto()
    FILE = auto()

    STEPS = auto()
    DT = auto()
    MODES = auto()
    NORMALIZED = auto()
    LAPLACIAN = auto()

    REMOVE_NODE = auto()
    REMOVE_EDGE = auto()
    REWEIGHT_EDGE = auto()
    BLOCK_PATH = auto()
    MUTATE_NODE = auto()
    FACTOR = auto()
    TARGET = auto()
    COMPARE = auto()

    FORMAT = auto()
    JSON = auto()
    MARKDOWN = auto()


KEYWORDS: Dict[str, TokenKind] = {
    "LOAD": TokenKind.LOAD,
    "GRAPH": TokenKind.GRAPH,
    "SEED": TokenKind.SEED,
    "DIFFUSE": TokenKind.DIFFUSE,
    "SPECTRAL": TokenKind.SPECTRAL,
    "COUNTERFACTUAL": TokenKind.COUNTERFACTUAL,
    "EXPORT": TokenKind.EXPORT,
    "LET": TokenKind.LET,
    "SHOW": TokenKind.SHOW,
    "DESCRIBE": TokenKind.DESCRIBE,
    "HELP": TokenKind.HELP,
    "AS": TokenKind.AS,
    "TO": TokenKind.TO,
    "FROM": TokenKind.FROM,
    "WITH": TokenKind.WITH,
    "USING": TokenKind.USING,
    "WHERE": TokenKind.WHERE,
    "AND": TokenKind.AND,
    "OR": TokenKind.OR,
    "CUTOFF": TokenKind.CUTOFF,
    "CHAIN": TokenKind.CHAIN,
    "WEIGHT": TokenKind.WEIGHT,
    "MODE": TokenKind.MODE,
    "MODEL": TokenKind.MODEL,
    "CONTACT": TokenKind.CONTACT,
    "DEMO": TokenKind.DEMO,
    "NPZ": TokenKind.NPZ,
    "CSV": TokenKind.CSV,
    "PDB": TokenKind.PDB,
    "NODE": TokenKind.NODE,
    "NODES": TokenKind.NODES,
    "EDGE": TokenKind.EDGE,
    "EDGES": TokenKind.EDGES,
    "PATH": TokenKind.PATH,
    "STRENGTH": TokenKind.STRENGTH,
    "VALUES": TokenKind.VALUES,
    "FILE": TokenKind.FILE,
    "STEPS": TokenKind.STEPS,
    "DT": TokenKind.DT,
    "MODES": TokenKind.MODES,
    "NORMALIZED": TokenKind.NORMALIZED,
    "LAPLACIAN": TokenKind.LAPLACIAN,
    "REMOVE_NODE": TokenKind.REMOVE_NODE,
    "REMOVE_EDGE": TokenKind.REMOVE_EDGE,
    "REWEIGHT_EDGE": TokenKind.REWEIGHT_EDGE,
    "BLOCK_PATH": TokenKind.BLOCK_PATH,
    "MUTATE_NODE": TokenKind.MUTATE_NODE,
    "FACTOR": TokenKind.FACTOR,
    "TARGET": TokenKind.TARGET,
    "COMPARE": TokenKind.COMPARE,
    "FORMAT": TokenKind.FORMAT,
    "JSON": TokenKind.JSON,
    "MARKDOWN": TokenKind.MARKDOWN,
    "TRUE": TokenKind.BOOLEAN,
    "FALSE": TokenKind.BOOLEAN,
}


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    lexeme: str
    value: Any
    span: SourceSpan

    def describe(self) -> str:
        if self.kind in {TokenKind.IDENT, TokenKind.STRING, TokenKind.INT, TokenKind.FLOAT, TokenKind.BOOLEAN}:
            return f"{self.kind.name}({self.value!r})"
        return self.kind.name


class Lexer:
    """Small hand-written lexer with comments, strings, numbers, and keywords."""

    def __init__(self, source: str):
        self.source = source
        self.length = len(source)
        self.index = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        while not self._at_end():
            ch = self._peek()
            if ch in " \t\r":
                self._advance()
            elif ch == "\n":
                self._emit_simple(TokenKind.NEWLINE)
                self._advance_newline()
            elif ch == ";":
                self._emit_simple(TokenKind.SEMICOLON)
                self._advance()
            elif ch == ",":
                self._emit_simple(TokenKind.COMMA)
                self._advance()
            elif ch == ":":
                self._emit_simple(TokenKind.COLON)
                self._advance()
            elif ch == "[":
                self._emit_simple(TokenKind.LBRACKET)
                self._advance()
            elif ch == "]":
                self._emit_simple(TokenKind.RBRACKET)
                self._advance()
            elif ch == "(":
                self._emit_simple(TokenKind.LPAREN)
                self._advance()
            elif ch == ")":
                self._emit_simple(TokenKind.RPAREN)
                self._advance()
            elif ch == "#":
                self._skip_comment()
            elif ch == "/" and self._peek_next() == "/":
                self._skip_comment()
            elif ch == '"' or ch == "'":
                self.tokens.append(self._read_string())
            elif ch.isdigit() or (ch in "+-" and self._peek_next().isdigit()):
                self.tokens.append(self._read_number())
            elif self._is_ident_start(ch):
                self.tokens.append(self._read_identifier_or_keyword())
            else:
                loc = self._location()
                span = SourceSpan(loc, loc)
                raise LexerError(f"Unexpected character {ch!r}", span, self.source)
        loc = self._location()
        self.tokens.append(Token(TokenKind.EOF, "", None, SourceSpan(loc, loc)))
        return self.tokens

    def _at_end(self) -> bool:
        return self.index >= self.length

    def _peek(self) -> str:
        return "\0" if self._at_end() else self.source[self.index]

    def _peek_next(self) -> str:
        j = self.index + 1
        return "\0" if j >= self.length else self.source[j]

    def _location(self) -> SourceLocation:
        return SourceLocation(self.line, self.column, self.index)

    def _advance(self) -> str:
        ch = self.source[self.index]
        self.index += 1
        self.column += 1
        return ch

    def _advance_newline(self) -> None:
        self.index += 1
        self.line += 1
        self.column = 1

    def _make_span(self, start: SourceLocation) -> SourceSpan:
        return SourceSpan(start, self._location())

    def _emit_simple(self, kind: TokenKind) -> None:
        start = self._location()
        ch = self._peek()
        end = SourceLocation(self.line, self.column + 1, self.index + 1)
        self.tokens.append(Token(kind, ch, ch, SourceSpan(start, end)))

    def _skip_comment(self) -> None:
        if self._peek() == "/" and self._peek_next() == "/":
            self._advance()
            self._advance()
        else:
            self._advance()
        while not self._at_end() and self._peek() != "\n":
            self._advance()

    def _read_string(self) -> Token:
        quote = self._peek()
        start = self._location()
        self._advance()
        chars: List[str] = []
        while not self._at_end():
            ch = self._peek()
            if ch == quote:
                self._advance()
                return Token(TokenKind.STRING, self.source[start.index:self.index], "".join(chars), self._make_span(start))
            if ch == "\n":
                raise LexerError("Unterminated string literal", self._make_span(start), self.source)
            if ch == "\\":
                self._advance()
                chars.append(self._read_escape(start))
            else:
                chars.append(self._advance())
        raise LexerError("Unterminated string literal", self._make_span(start), self.source)

    def _read_escape(self, start: SourceLocation) -> str:
        if self._at_end():
            raise LexerError("Unterminated string escape", self._make_span(start), self.source)
        ch = self._advance()
        escapes = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'"}
        if ch in escapes:
            return escapes[ch]
        raise LexerError(f"Unsupported escape sequence \\{ch}", self._make_span(start), self.source)

    def _read_number(self) -> Token:
        start = self._location()
        if self._peek() in "+-":
            self._advance()
        while self._peek().isdigit():
            self._advance()
        is_float = False
        if self._peek() == "." and self._peek_next().isdigit():
            is_float = True
            self._advance()
            while self._peek().isdigit():
                self._advance()
        if self._peek() in "eE":
            is_float = True
            self._advance()
            if self._peek() in "+-":
                self._advance()
            if not self._peek().isdigit():
                raise LexerError("Malformed exponent in number", self._make_span(start), self.source)
            while self._peek().isdigit():
                self._advance()
        text = self.source[start.index:self.index]
        try:
            if is_float:
                return Token(TokenKind.FLOAT, text, float(text), self._make_span(start))
            return Token(TokenKind.INT, text, int(text), self._make_span(start))
        except ValueError as exc:
            raise LexerError(f"Malformed number {text!r}", self._make_span(start), self.source) from exc

    def _read_identifier_or_keyword(self) -> Token:
        start = self._location()
        while self._is_ident_part(self._peek()):
            self._advance()
        text = self.source[start.index:self.index]
        upper = text.upper()
        kind = KEYWORDS.get(upper, TokenKind.IDENT)
        value: Any = text
        if kind == TokenKind.BOOLEAN:
            value = upper == "TRUE"
        elif kind != TokenKind.IDENT:
            value = upper
        return Token(kind, text, value, self._make_span(start))

    @staticmethod
    def _is_ident_start(ch: str) -> bool:
        return ch.isalpha() or ch == "_"

    @staticmethod
    def _is_ident_part(ch: str) -> bool:
        return ch.isalnum() or ch in "_.-"


# =============================================================================
# AST nodes
# =============================================================================


@dataclass
class ASTNode:
    span: SourceSpan


@dataclass
class Program(ASTNode):
    statements: List["Statement"]


class Statement(ASTNode):
    pass


class Expression(ASTNode):
    pass


@dataclass
class LiteralExpr(Expression):
    value: Any


@dataclass
class NameExpr(Expression):
    name: str


@dataclass
class ListExpr(Expression):
    items: List[Expression]


@dataclass
class LoadStatement(Statement):
    source: Expression
    alias: str
    loader: Optional[str] = None
    options: Dict[str, Expression] = field(default_factory=dict)


@dataclass
class GraphStatement(Statement):
    source_name: str
    alias: str
    cutoff: Optional[Expression] = None
    chain: Optional[Expression] = None
    weight_mode: Optional[Expression] = None
    contact_mode: Optional[Expression] = None


@dataclass
class SeedStatement(Statement):
    graph_name: str
    alias: str
    nodes: List[Expression]
    strength: Expression
    values_file: Optional[Expression] = None


@dataclass
class DiffuseStatement(Statement):
    signal_name: str
    alias: str
    steps: Expression
    dt: Expression
    model: Optional[Expression] = None


@dataclass
class SpectralStatement(Statement):
    graph_name: str
    alias: str
    modes: Expression
    normalized: bool = False
    laplacian_kind: str = "unnormalized"


@dataclass
class CounterfactualStatement(Statement):
    graph_name: str
    alias: str
    operation: str
    arguments: List[Expression]
    factor: Optional[Expression] = None
    target: Optional[Expression] = None
    compare_signal: Optional[str] = None


@dataclass
class ExportStatement(Statement):
    value_name: str
    path: Expression
    fmt: Optional[str] = None


@dataclass
class LetStatement(Statement):
    alias: str
    value: Expression


@dataclass
class ShowStatement(Statement):
    name: Optional[str] = None


@dataclass
class DescribeStatement(Statement):
    name: str


@dataclass
class HelpStatement(Statement):
    topic: Optional[str] = None


# =============================================================================
# Parser
# =============================================================================


class Parser:
    """Recursive-descent parser for RINetScript."""

    def __init__(self, tokens: Sequence[Token], source: str = ""):
        self.tokens = list(tokens)
        self.source = source
        self.pos = 0

    def parse(self) -> Program:
        start = self._peek().span.start
        statements: List[Statement] = []
        self._skip_separators()
        while not self._check(TokenKind.EOF):
            statements.append(self._statement())
            if self._check(TokenKind.EOF):
                break
            if not self._match(TokenKind.NEWLINE, TokenKind.SEMICOLON):
                tok = self._peek()
                raise ParserError(
                    f"Expected a newline or semicolon after statement, got {tok.describe()}",
                    tok.span,
                    self.source,
                )
            self._skip_separators()
        end = self._peek().span.end
        return Program(SourceSpan(start, end), statements)

    def _statement(self) -> Statement:
        tok = self._peek()
        if self._match(TokenKind.LOAD):
            return self._load_statement(tok)
        if self._match(TokenKind.GRAPH):
            return self._graph_statement(tok)
        if self._match(TokenKind.SEED):
            return self._seed_statement(tok)
        if self._match(TokenKind.DIFFUSE):
            return self._diffuse_statement(tok)
        if self._match(TokenKind.SPECTRAL):
            return self._spectral_statement(tok)
        if self._match(TokenKind.COUNTERFACTUAL):
            return self._counterfactual_statement(tok)
        if self._match(TokenKind.EXPORT):
            return self._export_statement(tok)
        if self._match(TokenKind.LET):
            return self._let_statement(tok)
        if self._match(TokenKind.SHOW):
            return self._show_statement(tok)
        if self._match(TokenKind.DESCRIBE):
            return self._describe_statement(tok)
        if self._match(TokenKind.HELP):
            return self._help_statement(tok)
        raise ParserError(f"Expected a statement, got {tok.describe()}", tok.span, self.source)

    def _load_statement(self, first: Token) -> LoadStatement:
        loader: Optional[str] = None
        if self._match(TokenKind.PDB, TokenKind.NPZ, TokenKind.CSV, TokenKind.DEMO):
            loader = self._previous().kind.name.lower()
        source = self._expression()
        options: Dict[str, Expression] = {}
        while self._match(TokenKind.CUTOFF, TokenKind.CHAIN, TokenKind.WEIGHT, TokenKind.MODE, TokenKind.CONTACT):
            key = self._previous().kind.name.lower()
            options[key] = self._expression()
        self._consume(TokenKind.AS, "Expected AS in LOAD statement")
        alias = self._consume_identifier("Expected alias after AS")
        return LoadStatement(SourceSpan(first.span.start, self._previous().span.end), source, alias, loader, options)

    def _graph_statement(self, first: Token) -> GraphStatement:
        source_name = self._consume_identifier("Expected source name after GRAPH")
        cutoff: Optional[Expression] = None
        chain: Optional[Expression] = None
        weight_mode: Optional[Expression] = None
        contact_mode: Optional[Expression] = None
        while self._check_any(TokenKind.CUTOFF, TokenKind.CHAIN, TokenKind.WEIGHT, TokenKind.MODE, TokenKind.CONTACT):
            if self._match(TokenKind.CUTOFF):
                cutoff = self._expression()
            elif self._match(TokenKind.CHAIN):
                chain = self._expression()
            elif self._match(TokenKind.WEIGHT, TokenKind.MODE):
                weight_mode = self._expression()
            elif self._match(TokenKind.CONTACT):
                contact_mode = self._expression()
        self._consume(TokenKind.AS, "Expected AS in GRAPH statement")
        alias = self._consume_identifier("Expected graph alias after AS")
        return GraphStatement(SourceSpan(first.span.start, self._previous().span.end), source_name, alias, cutoff, chain, weight_mode, contact_mode)

    def _seed_statement(self, first: Token) -> SeedStatement:
        graph_name = self._consume_identifier("Expected graph name after SEED")
        nodes: List[Expression] = []
        strength: Expression = LiteralExpr(first.span, 1.0)
        values_file: Optional[Expression] = None
        saw_seed_spec = False
        while self._check_any(TokenKind.NODE, TokenKind.NODES, TokenKind.STRENGTH, TokenKind.VALUES, TokenKind.FILE):
            if self._match(TokenKind.NODE):
                nodes.append(self._expression())
                saw_seed_spec = True
            elif self._match(TokenKind.NODES):
                nodes.extend(self._node_list())
                saw_seed_spec = True
            elif self._match(TokenKind.STRENGTH):
                strength = self._expression()
            elif self._match(TokenKind.VALUES):
                self._consume(TokenKind.FROM, "Expected FROM after VALUES")
                values_file = self._expression()
                saw_seed_spec = True
            elif self._match(TokenKind.FILE):
                values_file = self._expression()
                saw_seed_spec = True
        if not saw_seed_spec:
            tok = self._peek()
            raise ParserError("Expected NODE, NODES, or VALUES in SEED statement", tok.span, self.source)
        self._consume(TokenKind.AS, "Expected AS in SEED statement")
        alias = self._consume_identifier("Expected signal alias after AS")
        return SeedStatement(SourceSpan(first.span.start, self._previous().span.end), graph_name, alias, nodes, strength, values_file)

    def _diffuse_statement(self, first: Token) -> DiffuseStatement:
        signal_name = self._consume_identifier("Expected signal name after DIFFUSE")
        steps: Expression = LiteralExpr(first.span, 60)
        dt: Expression = LiteralExpr(first.span, 0.10)
        model: Optional[Expression] = None
        while self._check_any(TokenKind.STEPS, TokenKind.DT, TokenKind.MODEL, TokenKind.MODE):
            if self._match(TokenKind.STEPS):
                steps = self._expression()
            elif self._match(TokenKind.DT):
                dt = self._expression()
            elif self._match(TokenKind.MODEL, TokenKind.MODE):
                model = self._expression()
        self._consume(TokenKind.AS, "Expected AS in DIFFUSE statement")
        alias = self._consume_identifier("Expected output alias after AS")
        return DiffuseStatement(SourceSpan(first.span.start, self._previous().span.end), signal_name, alias, steps, dt, model)

    def _spectral_statement(self, first: Token) -> SpectralStatement:
        graph_name = self._consume_identifier("Expected graph name after SPECTRAL")
        modes: Expression = LiteralExpr(first.span, 20)
        normalized = False
        laplacian_kind = "unnormalized"
        while self._check_any(TokenKind.MODES, TokenKind.NORMALIZED, TokenKind.LAPLACIAN, TokenKind.MODE):
            if self._match(TokenKind.MODES):
                modes = self._expression()
            elif self._match(TokenKind.NORMALIZED):
                normalized = True
                laplacian_kind = "normalized"
            elif self._match(TokenKind.LAPLACIAN, TokenKind.MODE):
                expr = self._expression()
                if isinstance(expr, LiteralExpr):
                    laplacian_kind = str(expr.value)
                    normalized = laplacian_kind.lower() in {"normalized", "sym", "symmetric"}
                else:
                    raise ParserError("LAPLACIAN option must be a literal", expr.span, self.source)
        self._consume(TokenKind.AS, "Expected AS in SPECTRAL statement")
        alias = self._consume_identifier("Expected spectrum alias after AS")
        return SpectralStatement(SourceSpan(first.span.start, self._previous().span.end), graph_name, alias, modes, normalized, laplacian_kind)

    def _counterfactual_statement(self, first: Token) -> CounterfactualStatement:
        graph_name = self._consume_identifier("Expected graph name after COUNTERFACTUAL")
        op_tok = self._consume_any(
            "Expected counterfactual operation",
            TokenKind.REMOVE_NODE,
            TokenKind.REMOVE_EDGE,
            TokenKind.REWEIGHT_EDGE,
            TokenKind.BLOCK_PATH,
            TokenKind.MUTATE_NODE,
        )
        operation = op_tok.kind.name.lower()
        arguments: List[Expression] = []
        factor: Optional[Expression] = None
        target: Optional[Expression] = None
        compare_signal: Optional[str] = None
        if op_tok.kind in {TokenKind.REMOVE_NODE, TokenKind.MUTATE_NODE}:
            arguments.append(self._expression())
        elif op_tok.kind in {TokenKind.REMOVE_EDGE, TokenKind.REWEIGHT_EDGE}:
            arguments.extend(self._edge_pair())
        elif op_tok.kind == TokenKind.BLOCK_PATH:
            arguments.extend(self._node_list())
        while self._check_any(TokenKind.FACTOR, TokenKind.TARGET, TokenKind.COMPARE):
            if self._match(TokenKind.FACTOR):
                factor = self._expression()
            elif self._match(TokenKind.TARGET):
                target = self._expression()
            elif self._match(TokenKind.COMPARE):
                compare_signal = self._consume_identifier("Expected signal name after COMPARE")
        self._consume(TokenKind.AS, "Expected AS in COUNTERFACTUAL statement")
        alias = self._consume_identifier("Expected alias after AS")
        return CounterfactualStatement(
            SourceSpan(first.span.start, self._previous().span.end),
            graph_name,
            alias,
            operation,
            arguments,
            factor,
            target,
            compare_signal,
        )

    def _export_statement(self, first: Token) -> ExportStatement:
        value_name = self._consume_identifier("Expected value name after EXPORT")
        self._consume(TokenKind.TO, "Expected TO in EXPORT statement")
        path = self._expression()
        fmt: Optional[str] = None
        if self._match(TokenKind.FORMAT):
            fmt_tok = self._consume_any("Expected export format", TokenKind.IDENT, TokenKind.CSV, TokenKind.JSON, TokenKind.NPZ, TokenKind.MARKDOWN)
            fmt = str(fmt_tok.value).lower()
        return ExportStatement(SourceSpan(first.span.start, self._previous().span.end), value_name, path, fmt)

    def _let_statement(self, first: Token) -> LetStatement:
        alias = self._consume_identifier("Expected variable name after LET")
        if self._match(TokenKind.AS):
            value = self._expression()
        else:
            self._consume(TokenKind.TO, "Expected AS or TO in LET statement")
            value = self._expression()
        return LetStatement(SourceSpan(first.span.start, self._previous().span.end), alias, value)

    def _show_statement(self, first: Token) -> ShowStatement:
        if self._at_statement_end():
            return ShowStatement(SourceSpan(first.span.start, first.span.end), None)
        name = self._consume_identifier("Expected name after SHOW")
        return ShowStatement(SourceSpan(first.span.start, self._previous().span.end), name)

    def _describe_statement(self, first: Token) -> DescribeStatement:
        name = self._consume_identifier("Expected name after DESCRIBE")
        return DescribeStatement(SourceSpan(first.span.start, self._previous().span.end), name)

    def _help_statement(self, first: Token) -> HelpStatement:
        if self._at_statement_end():
            return HelpStatement(SourceSpan(first.span.start, first.span.end), None)
        tok = self._advance()
        return HelpStatement(SourceSpan(first.span.start, tok.span.end), str(tok.value))

    def _node_list(self) -> List[Expression]:
        if self._match(TokenKind.LBRACKET):
            nodes: List[Expression] = []
            if not self._check(TokenKind.RBRACKET):
                nodes.append(self._expression())
                while self._match(TokenKind.COMMA):
                    nodes.append(self._expression())
            self._consume(TokenKind.RBRACKET, "Expected ] after node list")
            return nodes
        first = self._expression()
        nodes = [first]
        while self._match(TokenKind.COMMA):
            nodes.append(self._expression())
        return nodes

    def _edge_pair(self) -> List[Expression]:
        if self._match(TokenKind.LPAREN):
            u = self._expression()
            self._consume(TokenKind.COMMA, "Expected comma between edge endpoints")
            v = self._expression()
            self._consume(TokenKind.RPAREN, "Expected ) after edge endpoints")
            return [u, v]
        u = self._expression()
        if self._match(TokenKind.COMMA):
            v = self._expression()
        else:
            v = self._expression()
        return [u, v]

    def _expression(self) -> Expression:
        tok = self._peek()
        if self._match(TokenKind.STRING, TokenKind.INT, TokenKind.FLOAT, TokenKind.BOOLEAN):
            return LiteralExpr(tok.span, tok.value)
        if self._match(TokenKind.IDENT):
            return NameExpr(tok.span, str(tok.value))
        if self._match(TokenKind.CSV, TokenKind.JSON, TokenKind.NPZ, TokenKind.PDB, TokenKind.DEMO, TokenKind.MARKDOWN):
            return LiteralExpr(tok.span, str(tok.value).lower())
        if self._match(TokenKind.LBRACKET):
            items: List[Expression] = []
            if not self._check(TokenKind.RBRACKET):
                items.append(self._expression())
                while self._match(TokenKind.COMMA):
                    items.append(self._expression())
            rbr = self._consume(TokenKind.RBRACKET, "Expected ] after list literal")
            return ListExpr(SourceSpan(tok.span.start, rbr.span.end), items)
        raise ParserError(f"Expected expression, got {tok.describe()}", tok.span, self.source)

    def _skip_separators(self) -> None:
        while self._match(TokenKind.NEWLINE, TokenKind.SEMICOLON):
            pass

    def _at_statement_end(self) -> bool:
        return self._check(TokenKind.NEWLINE) or self._check(TokenKind.SEMICOLON) or self._check(TokenKind.EOF)

    def _match(self, *kinds: TokenKind) -> bool:
        if self._check_any(*kinds):
            self._advance()
            return True
        return False

    def _consume(self, kind: TokenKind, message: str) -> Token:
        if self._check(kind):
            return self._advance()
        tok = self._peek()
        raise ParserError(f"{message}; got {tok.describe()}", tok.span, self.source)

    def _consume_any(self, message: str, *kinds: TokenKind) -> Token:
        if self._check_any(*kinds):
            return self._advance()
        tok = self._peek()
        expected = ", ".join(k.name for k in kinds)
        raise ParserError(f"{message}; expected one of {expected}, got {tok.describe()}", tok.span, self.source)

    def _consume_identifier(self, message: str) -> str:
        tok = self._consume(TokenKind.IDENT, message)
        return str(tok.value)

    def _check(self, kind: TokenKind) -> bool:
        return self._peek().kind == kind

    def _check_any(self, *kinds: TokenKind) -> bool:
        return self._peek().kind in kinds

    def _advance(self) -> Token:
        if not self._check(TokenKind.EOF):
            self.pos += 1
        return self._previous()

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _previous(self) -> Token:
        return self.tokens[self.pos - 1]


# =============================================================================
# Runtime value model
# =============================================================================


class ValueKind(Enum):
    PROTEIN = "protein"
    GRAPH = "graph"
    SIGNAL = "signal"
    DIFFUSION = "diffusion"
    SPECTRUM = "spectrum"
    COUNTERFACTUAL = "counterfactual"
    SCALAR = "scalar"
    ARRAY = "array"
    TABLE = "table"
    TEXT = "text"
    NONE = "none"


@dataclass
class RuntimeValue:
    kind: ValueKind
    name: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return f"{self.kind.value}"

    def to_jsonable(self) -> Any:
        return {"kind": self.kind.value, "name": self.name, "meta": self.meta}


@dataclass
class ProteinValue(RuntimeValue):
    path: str = ""
    loader: str = "pdb"
    options: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, path: str, loader: str = "pdb", options: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        super().__init__(ValueKind.PROTEIN, name=name, meta={"path": path, "loader": loader, "options": dict(options or {})})
        self.path = path
        self.loader = loader
        self.options = dict(options or {})

    def summary(self) -> str:
        return f"protein(path={self.path!r}, loader={self.loader})"


@dataclass
class GraphValue(RuntimeValue):
    bundle: GraphBundle = field(default_factory=lambda: synthetic_rin())

    def __init__(self, bundle: GraphBundle, name: Optional[str] = None):
        super().__init__(ValueKind.GRAPH, name=name or bundle.name, meta=dict(bundle.meta))
        self.bundle = bundle

    def summary(self) -> str:
        n = self.bundle.n
        edges = int(np.count_nonzero(np.triu(self.bundle.A, k=1)))
        return f"graph(name={self.bundle.name!r}, n={n}, edges={edges})"

    def to_jsonable(self) -> Any:
        return {"kind": self.kind.value, "name": self.name, "n": self.bundle.n, "meta": self.bundle.meta}


@dataclass
class SignalValue(RuntimeValue):
    graph_name: str = ""
    bundle: Optional[GraphBundle] = None
    vector: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __init__(self, graph_name: str, bundle: GraphBundle, vector: np.ndarray, name: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
        m = dict(meta or {})
        m.update({"graph": graph_name, "n": int(len(vector))})
        super().__init__(ValueKind.SIGNAL, name=name, meta=m)
        self.graph_name = graph_name
        self.bundle = bundle
        self.vector = np.asarray(vector, dtype=float)

    def summary(self) -> str:
        nz = int(np.count_nonzero(np.abs(self.vector) > 1e-15))
        return f"signal(graph={self.graph_name!r}, n={self.vector.size}, nonzero={nz}, norm={np.linalg.norm(self.vector):.6g})"

    def to_jsonable(self) -> Any:
        return {"kind": self.kind.value, "name": self.name, "graph": self.graph_name, "vector": self.vector.tolist(), "meta": self.meta}


@dataclass
class DiffusionValue(RuntimeValue):
    graph_name: str = ""
    vector: np.ndarray = field(default_factory=lambda: np.zeros(0))
    result_meta: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, graph_name: str, vector: np.ndarray, result_meta: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        meta = dict(result_meta or {})
        meta.update({"graph": graph_name, "n": int(len(vector))})
        super().__init__(ValueKind.DIFFUSION, name=name, meta=meta)
        self.graph_name = graph_name
        self.vector = np.asarray(vector, dtype=float)
        self.result_meta = dict(result_meta or {})

    def summary(self) -> str:
        return f"diffusion(graph={self.graph_name!r}, n={self.vector.size}, sum={np.sum(self.vector):.6g}, norm={np.linalg.norm(self.vector):.6g})"

    def to_jsonable(self) -> Any:
        return {"kind": self.kind.value, "name": self.name, "graph": self.graph_name, "state": self.vector.tolist(), "meta": self.meta}


@dataclass
class SpectrumValue(RuntimeValue):
    eigenvalues: np.ndarray = field(default_factory=lambda: np.zeros(0))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    laplacian_kind: str = "unnormalized"

    def __init__(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray, laplacian_kind: str = "unnormalized", name: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
        m = dict(meta or {})
        m.update({"modes": int(len(eigenvalues)), "laplacian_kind": laplacian_kind})
        super().__init__(ValueKind.SPECTRUM, name=name, meta=m)
        self.eigenvalues = np.asarray(eigenvalues, dtype=float)
        self.eigenvectors = np.asarray(eigenvectors, dtype=float)
        self.laplacian_kind = laplacian_kind

    def summary(self) -> str:
        if self.eigenvalues.size == 0:
            return "spectrum(empty)"
        lam0 = float(self.eigenvalues[0])
        lam1 = float(self.eigenvalues[1]) if self.eigenvalues.size > 1 else float("nan")
        return f"spectrum(modes={self.eigenvalues.size}, lambda0={lam0:.6g}, lambda1={lam1:.6g}, kind={self.laplacian_kind})"

    def to_jsonable(self) -> Any:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "eigenvalues": self.eigenvalues.tolist(),
            "eigenvectors": self.eigenvectors.tolist(),
            "laplacian_kind": self.laplacian_kind,
            "meta": self.meta,
        }


@dataclass
class CounterfactualValue(RuntimeValue):
    operation: str = ""
    graph_before: Optional[GraphBundle] = None
    graph_after: Optional[GraphBundle] = None
    report: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, operation: str, graph_before: GraphBundle, graph_after: GraphBundle, report: Dict[str, Any], name: Optional[str] = None):
        super().__init__(ValueKind.COUNTERFACTUAL, name=name, meta=dict(report))
        self.operation = operation
        self.graph_before = graph_before
        self.graph_after = graph_after
        self.report = dict(report)

    def summary(self) -> str:
        before_edges = int(np.count_nonzero(np.triu(self.graph_before.A, 1))) if self.graph_before is not None else 0
        after_edges = int(np.count_nonzero(np.triu(self.graph_after.A, 1))) if self.graph_after is not None else 0
        return f"counterfactual(operation={self.operation}, edges={before_edges}->{after_edges})"

    def to_jsonable(self) -> Any:
        return {"kind": self.kind.value, "name": self.name, "operation": self.operation, "report": self.report}


@dataclass
class ScalarValue(RuntimeValue):
    value: Union[int, float, bool, str] = 0

    def __init__(self, value: Union[int, float, bool, str], name: Optional[str] = None):
        super().__init__(ValueKind.SCALAR, name=name, meta={})
        self.value = value

    def summary(self) -> str:
        return f"scalar({self.value!r})"

    def to_jsonable(self) -> Any:
        return self.value


@dataclass
class ArrayValue(RuntimeValue):
    array: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __init__(self, array: np.ndarray, name: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
        super().__init__(ValueKind.ARRAY, name=name, meta=dict(meta or {}))
        self.array = np.asarray(array, dtype=float)

    def summary(self) -> str:
        return f"array(shape={self.array.shape}, min={np.min(self.array):.6g}, max={np.max(self.array):.6g})" if self.array.size else "array(empty)"

    def to_jsonable(self) -> Any:
        return {"kind": self.kind.value, "name": self.name, "shape": list(self.array.shape), "array": self.array.tolist(), "meta": self.meta}


@dataclass
class TextValue(RuntimeValue):
    text: str = ""

    def __init__(self, text: str, name: Optional[str] = None):
        super().__init__(ValueKind.TEXT, name=name, meta={})
        self.text = text

    def summary(self) -> str:
        one_line = self.text.replace("\n", " ")
        return one_line[:120] + ("..." if len(one_line) > 120 else "")

    def to_jsonable(self) -> Any:
        return self.text


# =============================================================================
# Environment and symbol handling
# =============================================================================


@dataclass
class SymbolRecord:
    name: str
    value: RuntimeValue
    defined_at: Optional[SourceSpan] = None

    @property
    def kind(self) -> ValueKind:
        return self.value.kind


class RuntimeEnvironment:
    """Mutable runtime store with helpful error messages."""

    def __init__(self):
        self._symbols: Dict[str, SymbolRecord] = {}

    def set(self, name: str, value: RuntimeValue, span: Optional[SourceSpan] = None) -> RuntimeValue:
        value.name = name
        self._symbols[name] = SymbolRecord(name, value, span)
        return value

    def get(self, name: str, span: Optional[SourceSpan] = None) -> RuntimeValue:
        if name not in self._symbols:
            raise RuntimeEvaluationError(f"Unknown symbol {name!r}", span)
        return self._symbols[name].value

    def require(self, name: str, kind: Union[ValueKind, Tuple[ValueKind, ...]], span: Optional[SourceSpan] = None) -> RuntimeValue:
        value = self.get(name, span)
        kinds = kind if isinstance(kind, tuple) else (kind,)
        if value.kind not in kinds:
            expected = ", ".join(k.value for k in kinds)
            raise RuntimeEvaluationError(f"Expected {name!r} to be {expected}, got {value.kind.value}", span)
        return value

    def has(self, name: str) -> bool:
        return name in self._symbols

    def names(self) -> List[str]:
        return sorted(self._symbols)

    def records(self) -> List[SymbolRecord]:
        return [self._symbols[k] for k in sorted(self._symbols)]

    def snapshot(self) -> Dict[str, Any]:
        return {name: rec.value.to_jsonable() for name, rec in self._symbols.items()}


# =============================================================================
# Semantic analyzer
# =============================================================================


@dataclass
class StaticSymbol:
    name: str
    kind: ValueKind
    span: SourceSpan


class SemanticAnalyzer:
    """Performs lightweight static checks before runtime execution."""

    def __init__(self):
        self.symbols: Dict[str, StaticSymbol] = {}

    def analyze(self, program: Program) -> None:
        for stmt in program.statements:
            self._analyze_statement(stmt)

    def _analyze_statement(self, stmt: Statement) -> None:
        if isinstance(stmt, LoadStatement):
            loader = (stmt.loader or self._infer_loader_from_literal(stmt.source)).lower()
            kind = ValueKind.GRAPH if loader in {"npz", "csv", "demo"} else ValueKind.PROTEIN
            self._define(stmt.alias, kind, stmt.span)
        elif isinstance(stmt, GraphStatement):
            self._require(stmt.source_name, (ValueKind.PROTEIN, ValueKind.GRAPH), stmt.span)
            self._define(stmt.alias, ValueKind.GRAPH, stmt.span)
        elif isinstance(stmt, SeedStatement):
            self._require(stmt.graph_name, ValueKind.GRAPH, stmt.span)
            self._define(stmt.alias, ValueKind.SIGNAL, stmt.span)
        elif isinstance(stmt, DiffuseStatement):
            self._require(stmt.signal_name, ValueKind.SIGNAL, stmt.span)
            self._define(stmt.alias, ValueKind.DIFFUSION, stmt.span)
        elif isinstance(stmt, SpectralStatement):
            self._require(stmt.graph_name, ValueKind.GRAPH, stmt.span)
            self._define(stmt.alias, ValueKind.SPECTRUM, stmt.span)
        elif isinstance(stmt, CounterfactualStatement):
            self._require(stmt.graph_name, ValueKind.GRAPH, stmt.span)
            if stmt.compare_signal:
                self._require(stmt.compare_signal, ValueKind.SIGNAL, stmt.span)
            self._define(stmt.alias, ValueKind.COUNTERFACTUAL, stmt.span)
        elif isinstance(stmt, ExportStatement):
            self._require_any(stmt.value_name, stmt.span)
        elif isinstance(stmt, LetStatement):
            self._define(stmt.alias, ValueKind.SCALAR, stmt.span)
        elif isinstance(stmt, DescribeStatement):
            self._require_any(stmt.name, stmt.span)
        elif isinstance(stmt, ShowStatement):
            if stmt.name is not None:
                self._require_any(stmt.name, stmt.span)
        elif isinstance(stmt, HelpStatement):
            return
        else:
            raise SemanticError(f"No semantic rule for {type(stmt).__name__}", stmt.span)

    def _define(self, name: str, kind: ValueKind, span: SourceSpan) -> None:
        if name in KEYWORDS:
            raise SemanticError(f"Cannot use reserved keyword {name!r} as a symbol", span)
        self.symbols[name] = StaticSymbol(name, kind, span)

    def _require(self, name: str, kinds: Union[ValueKind, Tuple[ValueKind, ...]], span: SourceSpan) -> None:
        if name not in self.symbols:
            raise SemanticError(f"Symbol {name!r} is used before it is defined", span)
        expected = kinds if isinstance(kinds, tuple) else (kinds,)
        actual = self.symbols[name].kind
        if actual not in expected:
            expected_text = ", ".join(k.value for k in expected)
            raise SemanticError(f"Symbol {name!r} must be {expected_text}, not {actual.value}", span)

    def _require_any(self, name: str, span: SourceSpan) -> None:
        if name not in self.symbols:
            raise SemanticError(f"Symbol {name!r} is used before it is defined", span)

    def _infer_loader_from_literal(self, expr: Expression) -> str:
        if isinstance(expr, LiteralExpr) and isinstance(expr.value, str):
            suffix = pathlib.Path(expr.value).suffix.lower()
            if suffix == ".npz":
                return "npz"
            if suffix == ".csv":
                return "csv"
            return "pdb"
        return "pdb"


# =============================================================================
# Execution hooks
# =============================================================================


@dataclass
class RINetExecutionHooks:
    """Override points for connecting RINetScript to advanced engines later."""

    load_pdb: Optional[Callable[..., GraphBundle]] = None
    load_npz: Optional[Callable[[str], GraphBundle]] = None
    load_csv: Optional[Callable[[str], GraphBundle]] = None
    make_demo: Optional[Callable[..., GraphBundle]] = None
    graph_from_protein: Optional[Callable[..., GraphBundle]] = None
    diffuse: Optional[Callable[..., ForwardResult]] = None
    spectral: Optional[Callable[..., SpectrumValue]] = None
    counterfactual: Optional[Callable[..., CounterfactualValue]] = None
    export: Optional[Callable[..., None]] = None


# =============================================================================
# Interpreter core
# =============================================================================


@dataclass
class ExecutionResult:
    program: Program
    environment: RuntimeEnvironment
    outputs: List[RuntimeValue]
    messages: List[str]

    def last(self) -> Optional[RuntimeValue]:
        return self.outputs[-1] if self.outputs else None


class RINetInterpreter:
    """Executes parsed RINetScript programs."""

    def __init__(self, hooks: Optional[RINetExecutionHooks] = None, env: Optional[RuntimeEnvironment] = None, cwd: Optional[Union[str, os.PathLike[str]]] = None):
        self.hooks = hooks or RINetExecutionHooks()
        self.env = env or RuntimeEnvironment()
        self.cwd = pathlib.Path(cwd or os.getcwd())
        self.messages: List[str] = []

    def execute(self, program: Program) -> ExecutionResult:
        outputs: List[RuntimeValue] = []
        for stmt in program.statements:
            out = self._execute_statement(stmt)
            if out is not None:
                outputs.append(out)
        return ExecutionResult(program, self.env, outputs, list(self.messages))

    def _execute_statement(self, stmt: Statement) -> Optional[RuntimeValue]:
        try:
            if isinstance(stmt, LoadStatement):
                return self._execute_load(stmt)
            if isinstance(stmt, GraphStatement):
                return self._execute_graph(stmt)
            if isinstance(stmt, SeedStatement):
                return self._execute_seed(stmt)
            if isinstance(stmt, DiffuseStatement):
                return self._execute_diffuse(stmt)
            if isinstance(stmt, SpectralStatement):
                return self._execute_spectral(stmt)
            if isinstance(stmt, CounterfactualStatement):
                return self._execute_counterfactual(stmt)
            if isinstance(stmt, ExportStatement):
                return self._execute_export(stmt)
            if isinstance(stmt, LetStatement):
                return self._execute_let(stmt)
            if isinstance(stmt, ShowStatement):
                return self._execute_show(stmt)
            if isinstance(stmt, DescribeStatement):
                return self._execute_describe(stmt)
            if isinstance(stmt, HelpStatement):
                return self._execute_help(stmt)
            raise RuntimeEvaluationError(f"Unsupported statement {type(stmt).__name__}", stmt.span)
        except RINetLanguageError:
            raise
        except Exception as exc:
            raise RuntimeEvaluationError(str(exc), stmt.span) from exc

    def _execute_load(self, stmt: LoadStatement) -> RuntimeValue:
        source_value = self._eval_expr(stmt.source)
        options = {k: self._eval_expr(v) for k, v in stmt.options.items()}
        loader = (stmt.loader or self._infer_loader(source_value)).lower()
        if loader == "demo":
            n = int(options.get("n", options.get("nodes", 60)))
            seed = int(options.get("seed", 0))
            maker = self.hooks.make_demo or synthetic_rin
            bundle = maker(n=n, seed=seed)
            return self.env.set(stmt.alias, GraphValue(bundle), stmt.span)
        if not isinstance(source_value, str):
            raise RuntimeEvaluationError("LOAD source must be a path string unless using LOAD DEMO", stmt.source.span)
        path = str(self._resolve_path(source_value))
        if loader == "npz":
            bundle = (self.hooks.load_npz or bundle_from_npz)(path)
            return self.env.set(stmt.alias, GraphValue(bundle), stmt.span)
        if loader == "csv":
            if self.hooks.load_csv is not None:
                bundle = self.hooks.load_csv(path)
            else:
                A = adjacency_from_csv(path)
                bundle = bundle_from_adjacency(A, name=pathlib.Path(path).stem, meta={"source": "csv_adjacency", "path": path})
            return self.env.set(stmt.alias, GraphValue(bundle), stmt.span)
        if loader == "pdb":
            value = ProteinValue(path, loader="pdb", options=options)
            return self.env.set(stmt.alias, value, stmt.span)
        raise RuntimeEvaluationError(f"Unknown LOAD loader {loader!r}", stmt.span)

    def _execute_graph(self, stmt: GraphStatement) -> GraphValue:
        source = self.env.require(stmt.source_name, (ValueKind.PROTEIN, ValueKind.GRAPH), stmt.span)
        if isinstance(source, GraphValue):
            bundle = GraphBundle(A=source.bundle.A.copy(), meta=dict(source.bundle.meta), name=source.bundle.name)
            return self.env.set(stmt.alias, GraphValue(bundle), stmt.span)  # type: ignore[return-value]
        if not isinstance(source, ProteinValue):
            raise RuntimeEvaluationError("GRAPH source must be a protein or graph", stmt.span)
        cutoff = float(self._eval_expr(stmt.cutoff)) if stmt.cutoff is not None else float(source.options.get("cutoff", 8.0))
        chain = self._eval_expr(stmt.chain) if stmt.chain is not None else source.options.get("chain")
        weight_mode = self._eval_expr(stmt.weight_mode) if stmt.weight_mode is not None else source.options.get("weight", source.options.get("mode", "binary"))
        contact_mode = self._eval_expr(stmt.contact_mode) if stmt.contact_mode is not None else source.options.get("contact")
        if self.hooks.graph_from_protein is not None:
            bundle = self.hooks.graph_from_protein(source.path, cutoff=cutoff, chain=chain, weight_mode=weight_mode, contact_mode=contact_mode)
        else:
            # Current pdb_to_rin accepts CA contact construction and ignores future contact_mode.
            bundle = pdb_to_rin(source.path, cutoff=cutoff, chain=chain, weight_mode=str(weight_mode))
            if contact_mode is not None:
                bundle.meta["contact_mode"] = str(contact_mode)
        return self.env.set(stmt.alias, GraphValue(bundle), stmt.span)  # type: ignore[return-value]

    def _execute_seed(self, stmt: SeedStatement) -> SignalValue:
        graph = self.env.require(stmt.graph_name, ValueKind.GRAPH, stmt.span)
        assert isinstance(graph, GraphValue)
        n = graph.bundle.n
        x = np.zeros(n, dtype=float)
        strength = float(self._eval_expr(stmt.strength))
        if stmt.values_file is not None:
            path = str(self._resolve_path(str(self._eval_expr(stmt.values_file))))
            vec = np.loadtxt(path, delimiter=",")
            vec = np.asarray(vec, dtype=float).reshape(-1)
            if vec.size != n:
                raise RuntimeEvaluationError(f"Signal file has length {vec.size}, expected graph size {n}", stmt.values_file.span)
            x = vec.copy()
        for node_expr in stmt.nodes:
            idx = int(self._eval_expr(node_expr))
            if idx < 0 or idx >= n:
                raise RuntimeEvaluationError(f"Seed node {idx} is out of bounds for graph with {n} nodes", node_expr.span)
            x[idx] = strength
        meta = {"seed_nodes": [int(self._eval_expr(e)) for e in stmt.nodes], "seed_strength": strength}
        return self.env.set(stmt.alias, SignalValue(stmt.graph_name, graph.bundle, x, meta=meta), stmt.span)  # type: ignore[return-value]

    def _execute_diffuse(self, stmt: DiffuseStatement) -> DiffusionValue:
        signal = self.env.require(stmt.signal_name, ValueKind.SIGNAL, stmt.span)
        assert isinstance(signal, SignalValue)
        steps = int(self._eval_expr(stmt.steps))
        dt = float(self._eval_expr(stmt.dt))
        model = str(self._eval_expr(stmt.model)) if stmt.model is not None else "diffusion"
        if self.hooks.diffuse is not None:
            out = self.hooks.diffuse(signal.bundle, signal.vector, steps=steps, dt=dt, model=model)
            if isinstance(out, ForwardResult):
                state = out.state
                meta = out.meta
            elif isinstance(out, DiffusionValue):
                return self.env.set(stmt.alias, out, stmt.span)  # type: ignore[return-value]
            else:
                state = np.asarray(out, dtype=float)
                meta = {"steps": steps, "dt": dt, "model": model}
        else:
            dyn = make_model(model)
            dyn_out = dyn.run(signal.bundle.A, x0=signal.vector, steps=steps, dt=dt)
            state = dyn_out.state
            meta = {**dyn_out.meta, "input_signal": stmt.signal_name, "graph": signal.graph_name}
        return self.env.set(stmt.alias, DiffusionValue(signal.graph_name, state, meta), stmt.span)  # type: ignore[return-value]

    def _execute_spectral(self, stmt: SpectralStatement) -> SpectrumValue:
        graph = self.env.require(stmt.graph_name, ValueKind.GRAPH, stmt.span)
        assert isinstance(graph, GraphValue)
        modes = int(self._eval_expr(stmt.modes))
        if modes <= 0:
            raise RuntimeEvaluationError("SPECTRAL MODES must be positive", stmt.modes.span)
        if self.hooks.spectral is not None:
            value = self.hooks.spectral(graph.bundle, modes=modes, normalized=stmt.normalized, laplacian_kind=stmt.laplacian_kind)
            return self.env.set(stmt.alias, value, stmt.span)  # type: ignore[return-value]
        eigenvalues, eigenvectors = default_spectral_decomposition(graph.bundle.A, modes=modes, kind=stmt.laplacian_kind)
        meta = {"graph": stmt.graph_name, "modes_requested": modes, "normalized": stmt.normalized}
        return self.env.set(stmt.alias, SpectrumValue(eigenvalues, eigenvectors, stmt.laplacian_kind, meta=meta), stmt.span)  # type: ignore[return-value]

    def _execute_counterfactual(self, stmt: CounterfactualStatement) -> CounterfactualValue:
        graph = self.env.require(stmt.graph_name, ValueKind.GRAPH, stmt.span)
        assert isinstance(graph, GraphValue)
        if self.hooks.counterfactual is not None:
            args = [self._eval_expr(x) for x in stmt.arguments]
            value = self.hooks.counterfactual(graph.bundle, operation=stmt.operation, args=args, factor=self._eval_optional(stmt.factor), target=self._eval_optional(stmt.target), compare_signal=stmt.compare_signal)
            return self.env.set(stmt.alias, value, stmt.span)  # type: ignore[return-value]
        before = graph.bundle
        after, report = apply_default_counterfactual(before, stmt.operation, [self._eval_expr(x) for x in stmt.arguments], factor=self._eval_optional(stmt.factor), target=self._eval_optional(stmt.target))
        if stmt.compare_signal:
            signal = self.env.require(stmt.compare_signal, ValueKind.SIGNAL, stmt.span)
            assert isinstance(signal, SignalValue)
            before_state = make_model("diffusion").run(before.A, signal.vector, steps=60, dt=0.1).state
            after_signal = resize_signal_for_counterfactual(signal.vector, report)
            after_state = make_model("diffusion").run(after.A, after_signal, steps=60, dt=0.1).state
            report["diffusion_compare"] = compare_vectors(before_state, after_state, report)
        return self.env.set(stmt.alias, CounterfactualValue(stmt.operation, before, after, report), stmt.span)  # type: ignore[return-value]

    def _execute_export(self, stmt: ExportStatement) -> RuntimeValue:
        value = self.env.get(stmt.value_name, stmt.span)
        path_value = self._eval_expr(stmt.path)
        if not isinstance(path_value, str):
            raise RuntimeEvaluationError("EXPORT path must be a string", stmt.path.span)
        path = self._resolve_path(path_value)
        fmt = stmt.fmt or infer_export_format(path)
        if self.hooks.export is not None:
            self.hooks.export(value, str(path), fmt=fmt)
        else:
            export_value(value, path, fmt)
        msg = TextValue(f"Exported {stmt.value_name} to {path}")
        self.messages.append(msg.text)
        return msg

    def _execute_let(self, stmt: LetStatement) -> ScalarValue:
        value = self._eval_expr(stmt.value)
        return self.env.set(stmt.alias, ScalarValue(value), stmt.span)  # type: ignore[return-value]

    def _execute_show(self, stmt: ShowStatement) -> TextValue:
        if stmt.name is None:
            lines = [f"{rec.name}: {rec.value.summary()}" for rec in self.env.records()]
            text = "\n".join(lines) if lines else "No symbols defined."
        else:
            value = self.env.get(stmt.name, stmt.span)
            text = value.summary()
        self.messages.append(text)
        return TextValue(text)

    def _execute_describe(self, stmt: DescribeStatement) -> TextValue:
        value = self.env.get(stmt.name, stmt.span)
        text = describe_value(value)
        self.messages.append(text)
        return TextValue(text)

    def _execute_help(self, stmt: HelpStatement) -> TextValue:
        text = help_text(stmt.topic)
        self.messages.append(text)
        return TextValue(text)

    def _eval_expr(self, expr: Optional[Expression]) -> Any:
        if expr is None:
            return None
        if isinstance(expr, LiteralExpr):
            return expr.value
        if isinstance(expr, NameExpr):
            value = self.env.get(expr.name, expr.span)
            if isinstance(value, ScalarValue):
                return value.value
            return value
        if isinstance(expr, ListExpr):
            return [self._eval_expr(x) for x in expr.items]
        raise RuntimeEvaluationError(f"Unsupported expression {type(expr).__name__}", expr.span)

    def _eval_optional(self, expr: Optional[Expression]) -> Any:
        return None if expr is None else self._eval_expr(expr)

    def _resolve_path(self, path: str) -> pathlib.Path:
        p = pathlib.Path(path).expanduser()
        if not p.is_absolute():
            p = self.cwd / p
        return p

    def _infer_loader(self, source_value: Any) -> str:
        if isinstance(source_value, str):
            suffix = pathlib.Path(source_value).suffix.lower()
            if suffix == ".npz":
                return "npz"
            if suffix == ".csv":
                return "csv"
        return "pdb"


# =============================================================================
# Numerical fallback implementations for hooks that do not exist yet
# =============================================================================


def adjacency_to_laplacian(A: np.ndarray, kind: str = "unnormalized") -> np.ndarray:
    A = np.asarray(A, dtype=float)
    deg = np.sum(A, axis=1)
    L = np.diag(deg) - A
    key = str(kind).lower()
    if key in {"unnormalized", "combinatorial", "plain"}:
        return L
    if key in {"normalized", "sym", "symmetric"}:
        inv_sqrt = np.zeros_like(deg)
        mask = deg > 0
        inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
        D_inv_sqrt = np.diag(inv_sqrt)
        return D_inv_sqrt @ L @ D_inv_sqrt
    if key in {"random_walk", "rw", "walk"}:
        inv = np.zeros_like(deg)
        mask = deg > 0
        inv[mask] = 1.0 / deg[mask]
        return np.diag(inv) @ L
    raise RuntimeEvaluationError(f"Unknown Laplacian kind {kind!r}")


def default_spectral_decomposition(A: np.ndarray, modes: int = 20, kind: str = "unnormalized") -> Tuple[np.ndarray, np.ndarray]:
    L = adjacency_to_laplacian(A, kind=kind)
    if str(kind).lower() in {"random_walk", "rw", "walk"}:
        vals, vecs = np.linalg.eig(L)
        vals = vals.real
        vecs = vecs.real
    else:
        L = 0.5 * (L + L.T)
        vals, vecs = np.linalg.eigh(L)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    k = min(int(modes), vals.size)
    return vals[:k], vecs[:, :k]


def apply_default_counterfactual(bundle: GraphBundle, operation: str, args: Sequence[Any], factor: Any = None, target: Any = None) -> Tuple[GraphBundle, Dict[str, Any]]:
    A = np.asarray(bundle.A, dtype=float).copy()
    n_before = int(A.shape[0])
    operation = operation.lower()
    report: Dict[str, Any] = {
        "operation": operation,
        "graph_name": bundle.name,
        "n_before": n_before,
        "edges_before": int(np.count_nonzero(np.triu(A, 1))),
        "args": list(map(_json_sanitize, args)),
    }
    if operation == "remove_node":
        node = int(args[0])
        _check_node(node, A.shape[0])
        keep = [i for i in range(A.shape[0]) if i != node]
        B = A[np.ix_(keep, keep)]
        meta = dict(bundle.meta)
        meta["counterfactual_removed_node"] = node
        meta["original_to_counterfactual_index"] = {str(old): int(new) for new, old in enumerate(keep)}
        report.update({"removed_node": node, "n_after": int(B.shape[0]), "index_map": meta["original_to_counterfactual_index"]})
    elif operation == "remove_edge":
        u, v = int(args[0]), int(args[1])
        _check_node(u, A.shape[0])
        _check_node(v, A.shape[0])
        old_weight = float(A[u, v])
        A[u, v] = A[v, u] = 0.0
        B = A
        meta = dict(bundle.meta)
        meta["counterfactual_removed_edge"] = [u, v]
        report.update({"removed_edge": [u, v], "old_weight": old_weight, "n_after": int(B.shape[0])})
    elif operation == "reweight_edge":
        u, v = int(args[0]), int(args[1])
        _check_node(u, A.shape[0])
        _check_node(v, A.shape[0])
        f = float(0.5 if factor is None else factor)
        old_weight = float(A[u, v])
        A[u, v] = A[v, u] = old_weight * f
        B = A
        meta = dict(bundle.meta)
        meta["counterfactual_reweighted_edge"] = [u, v]
        meta["counterfactual_factor"] = f
        report.update({"reweighted_edge": [u, v], "old_weight": old_weight, "new_weight": float(A[u, v]), "factor": f, "n_after": int(B.shape[0])})
    elif operation == "mutate_node":
        node = int(args[0])
        _check_node(node, A.shape[0])
        f = float(0.5 if factor is None else factor)
        old_strength = float(np.sum(A[node, :]))
        A[node, :] *= f
        A[:, node] *= f
        A[node, node] = 0.0
        B = A
        meta = dict(bundle.meta)
        meta["counterfactual_mutated_node"] = node
        meta["counterfactual_factor"] = f
        report.update({"mutated_node": node, "old_degree_weight": old_strength, "new_degree_weight": float(np.sum(A[node, :])), "factor": f, "n_after": int(B.shape[0])})
    elif operation == "block_path":
        path_nodes = [int(x) for x in args]
        for node in path_nodes:
            _check_node(node, A.shape[0])
        removed_edges: List[List[int]] = []
        for u, v in zip(path_nodes, path_nodes[1:]):
            if A[u, v] != 0:
                removed_edges.append([u, v])
            A[u, v] = A[v, u] = 0.0
        B = A
        meta = dict(bundle.meta)
        meta["counterfactual_blocked_path"] = path_nodes
        report.update({"blocked_path": path_nodes, "removed_path_edges": removed_edges, "n_after": int(B.shape[0])})
    else:
        raise RuntimeEvaluationError(f"Unsupported counterfactual operation {operation!r}")
    report["edges_after"] = int(np.count_nonzero(np.triu(B, 1)))
    report["total_weight_before"] = float(np.sum(bundle.A) / 2.0)
    report["total_weight_after"] = float(np.sum(B) / 2.0)
    after = GraphBundle(A=B, meta=meta, name=f"{bundle.name}|cf:{operation}")
    return after, report


def _check_node(node: int, n: int) -> None:
    if node < 0 or node >= n:
        raise RuntimeEvaluationError(f"Node {node} is out of bounds for graph with {n} nodes")


def resize_signal_for_counterfactual(signal: np.ndarray, report: Mapping[str, Any]) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if "index_map" in report:
        pairs = sorted(((int(old), int(new)) for old, new in report["index_map"].items()), key=lambda x: x[1])
        return np.asarray([signal[old] for old, _new in pairs], dtype=float)
    return signal.copy()


def compare_vectors(before: np.ndarray, after: np.ndarray, report: Mapping[str, Any]) -> Dict[str, Any]:
    before = np.asarray(before, dtype=float).reshape(-1)
    after = np.asarray(after, dtype=float).reshape(-1)
    if before.size != after.size and "index_map" in report:
        pairs = sorted(((int(old), int(new)) for old, new in report["index_map"].items()), key=lambda x: x[1])
        before_cmp = np.asarray([before[old] for old, _new in pairs], dtype=float)
        after_cmp = after
    else:
        k = min(before.size, after.size)
        before_cmp = before[:k]
        after_cmp = after[:k]
    delta = after_cmp - before_cmp
    denom = np.linalg.norm(before_cmp) + 1e-12
    return {
        "n_compared": int(delta.size),
        "l2_delta": float(np.linalg.norm(delta)),
        "relative_l2_delta": float(np.linalg.norm(delta) / denom),
        "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "sum_before": float(np.sum(before_cmp)),
        "sum_after": float(np.sum(after_cmp)),
    }


# =============================================================================
# Export and description helpers
# =============================================================================


def infer_export_format(path: Union[str, pathlib.Path]) -> str:
    suffix = pathlib.Path(path).suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".npz":
        return "npz"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    return "csv"


def export_value(value: RuntimeValue, path: Union[str, pathlib.Path], fmt: str) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if isinstance(value, GraphValue) and fmt == "npz":
        bundle_to_npz(value.bundle, str(path))
        return
    if fmt == "json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(value.to_jsonable(), f, indent=2, sort_keys=True)
        return
    if fmt == "markdown":
        with path.open("w", encoding="utf-8") as f:
            f.write(markdown_for_value(value))
        return
    if fmt == "csv":
        export_value_csv(value, path)
        return
    raise RuntimeEvaluationError(f"Unsupported export format {fmt!r}")


def export_value_csv(value: RuntimeValue, path: pathlib.Path) -> None:
    if isinstance(value, GraphValue):
        np.savetxt(path, value.bundle.A, delimiter=",")
    elif isinstance(value, SignalValue):
        write_vector_csv(path, value.vector, header=["node", "signal"])
    elif isinstance(value, DiffusionValue):
        write_vector_csv(path, value.vector, header=["node", "state"])
    elif isinstance(value, SpectrumValue):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["mode", "eigenvalue"])
            for i, val in enumerate(value.eigenvalues):
                writer.writerow([i, float(val)])
    elif isinstance(value, CounterfactualValue):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for k, v in value.report.items():
                writer.writerow([k, json.dumps(_json_sanitize(v))])
    elif isinstance(value, ArrayValue):
        np.savetxt(path, value.array, delimiter=",")
    elif isinstance(value, ScalarValue):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["value"])
            writer.writerow([value.value])
    elif isinstance(value, TextValue):
        path.write_text(value.text, encoding="utf-8")
    else:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["json"])
            writer.writerow([json.dumps(value.to_jsonable())])


def write_vector_csv(path: pathlib.Path, vector: np.ndarray, header: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, x in enumerate(np.asarray(vector).reshape(-1)):
            writer.writerow([i, float(x)])


def markdown_for_value(value: RuntimeValue) -> str:
    lines = [f"# {value.name or value.kind.value}", "", f"**Kind:** `{value.kind.value}`", "", "## Summary", "", value.summary(), ""]
    if isinstance(value, GraphValue):
        lines += ["## Graph", "", f"- Nodes: {value.bundle.n}", f"- Edges: {int(np.count_nonzero(np.triu(value.bundle.A, 1)))}", ""]
    elif isinstance(value, SpectrumValue):
        lines += ["## Eigenvalues", "", "| mode | eigenvalue |", "|---:|---:|"]
        for i, val in enumerate(value.eigenvalues):
            lines.append(f"| {i} | {float(val):.12g} |")
        lines.append("")
    elif isinstance(value, CounterfactualValue):
        lines += ["## Report", ""]
        for k, v in value.report.items():
            lines.append(f"- **{k}:** `{json.dumps(_json_sanitize(v))}`")
        lines.append("")
    else:
        lines += ["## JSON", "", "```json", json.dumps(value.to_jsonable(), indent=2, sort_keys=True), "```", ""]
    return "\n".join(lines)


def describe_value(value: RuntimeValue) -> str:
    if isinstance(value, ProteinValue):
        return json.dumps(value.to_jsonable(), indent=2, sort_keys=True)
    if isinstance(value, GraphValue):
        A = value.bundle.A
        deg = np.sum(A > 0, axis=1)
        payload = {
            "kind": "graph",
            "name": value.bundle.name,
            "nodes": value.bundle.n,
            "edges": int(np.count_nonzero(np.triu(A, 1))),
            "weighted_degree_min": float(np.min(np.sum(A, axis=1))) if value.bundle.n else 0.0,
            "weighted_degree_max": float(np.max(np.sum(A, axis=1))) if value.bundle.n else 0.0,
            "degree_min": int(np.min(deg)) if deg.size else 0,
            "degree_max": int(np.max(deg)) if deg.size else 0,
            "meta": value.bundle.meta,
        }
        return json.dumps(payload, indent=2, sort_keys=True)
    return json.dumps(value.to_jsonable(), indent=2, sort_keys=True)


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, RuntimeValue):
        return obj.to_jsonable()
    return obj


# =============================================================================
# Script formatter
# =============================================================================


class ScriptFormatter:
    """Converts parsed AST back into canonical RINetScript."""

    def format(self, program: Program) -> str:
        return "\n".join(self.format_statement(stmt) for stmt in program.statements)

    def format_statement(self, stmt: Statement) -> str:
        if isinstance(stmt, LoadStatement):
            loader = f" {stmt.loader.upper()}" if stmt.loader else ""
            opts = "".join(f" {k.upper()} {self.expr(v)}" for k, v in stmt.options.items())
            return f"LOAD{loader} {self.expr(stmt.source)}{opts} AS {stmt.alias}"
        if isinstance(stmt, GraphStatement):
            parts = ["GRAPH", stmt.source_name]
            if stmt.cutoff is not None:
                parts += ["CUTOFF", self.expr(stmt.cutoff)]
            if stmt.chain is not None:
                parts += ["CHAIN", self.expr(stmt.chain)]
            if stmt.weight_mode is not None:
                parts += ["WEIGHT", self.expr(stmt.weight_mode)]
            if stmt.contact_mode is not None:
                parts += ["CONTACT", self.expr(stmt.contact_mode)]
            parts += ["AS", stmt.alias]
            return " ".join(parts)
        if isinstance(stmt, SeedStatement):
            parts = ["SEED", stmt.graph_name]
            if stmt.nodes:
                if len(stmt.nodes) == 1:
                    parts += ["NODE", self.expr(stmt.nodes[0])]
                else:
                    parts += ["NODES", "[" + ", ".join(self.expr(x) for x in stmt.nodes) + "]"]
            if stmt.values_file is not None:
                parts += ["VALUES", "FROM", self.expr(stmt.values_file)]
            parts += ["STRENGTH", self.expr(stmt.strength), "AS", stmt.alias]
            return " ".join(parts)
        if isinstance(stmt, DiffuseStatement):
            parts = ["DIFFUSE", stmt.signal_name, "STEPS", self.expr(stmt.steps), "DT", self.expr(stmt.dt)]
            if stmt.model is not None:
                parts += ["MODEL", self.expr(stmt.model)]
            parts += ["AS", stmt.alias]
            return " ".join(parts)
        if isinstance(stmt, SpectralStatement):
            parts = ["SPECTRAL", stmt.graph_name, "MODES", self.expr(stmt.modes)]
            if stmt.laplacian_kind != "unnormalized":
                parts += ["LAPLACIAN", self._quote_if_needed(stmt.laplacian_kind)]
            parts += ["AS", stmt.alias]
            return " ".join(parts)
        if isinstance(stmt, CounterfactualStatement):
            parts = ["COUNTERFACTUAL", stmt.graph_name, stmt.operation.upper()]
            parts += [self.expr(x) for x in stmt.arguments]
            if stmt.factor is not None:
                parts += ["FACTOR", self.expr(stmt.factor)]
            if stmt.target is not None:
                parts += ["TARGET", self.expr(stmt.target)]
            if stmt.compare_signal:
                parts += ["COMPARE", stmt.compare_signal]
            parts += ["AS", stmt.alias]
            return " ".join(parts)
        if isinstance(stmt, ExportStatement):
            tail = f" FORMAT {stmt.fmt}" if stmt.fmt else ""
            return f"EXPORT {stmt.value_name} TO {self.expr(stmt.path)}{tail}"
        if isinstance(stmt, LetStatement):
            return f"LET {stmt.alias} AS {self.expr(stmt.value)}"
        if isinstance(stmt, ShowStatement):
            return f"SHOW {stmt.name}" if stmt.name else "SHOW"
        if isinstance(stmt, DescribeStatement):
            return f"DESCRIBE {stmt.name}"
        if isinstance(stmt, HelpStatement):
            return f"HELP {stmt.topic}" if stmt.topic else "HELP"
        return repr(stmt)

    def expr(self, expr: Expression) -> str:
        if isinstance(expr, LiteralExpr):
            if isinstance(expr.value, str):
                return self._quote_if_needed(expr.value)
            if isinstance(expr.value, bool):
                return "TRUE" if expr.value else "FALSE"
            return str(expr.value)
        if isinstance(expr, NameExpr):
            return expr.name
        if isinstance(expr, ListExpr):
            return "[" + ", ".join(self.expr(x) for x in expr.items) + "]"
        return repr(expr)

    def _quote_if_needed(self, value: str) -> str:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_.-]*$", value) and value.upper() not in KEYWORDS:
            return value
        return json.dumps(value)


# =============================================================================
# Convenience API
# =============================================================================


def parse_script(source: str) -> Program:
    tokens = Lexer(source).tokenize()
    return Parser(tokens, source).parse()


def analyze_script(source: str) -> Program:
    program = parse_script(source)
    SemanticAnalyzer().analyze(program)
    return program


def run_script(source: str, hooks: Optional[RINetExecutionHooks] = None, env: Optional[RuntimeEnvironment] = None, cwd: Optional[Union[str, os.PathLike[str]]] = None, analyze: bool = True) -> ExecutionResult:
    program = parse_script(source)
    if analyze:
        SemanticAnalyzer().analyze(program)
    interpreter = RINetInterpreter(hooks=hooks, env=env, cwd=cwd)
    return interpreter.execute(program)


def format_script(source: str) -> str:
    return ScriptFormatter().format(parse_script(source))


def tokenize_script(source: str) -> List[Token]:
    return Lexer(source).tokenize()


# =============================================================================
# Help text and examples
# =============================================================================


EXAMPLE_BASIC = '''\
LOAD DEMO "synthetic" AS protein_graph
SEED protein_graph NODE 5 STRENGTH 1.0 AS signal
DIFFUSE signal STEPS 80 DT 0.05 AS output
SPECTRAL protein_graph MODES 12 AS spectrum
COUNTERFACTUAL protein_graph REMOVE_NODE 10 AS cf
EXPORT output TO "output.csv"
'''

EXAMPLE_PDB = '''\
LOAD "protein.pdb" AS protein
GRAPH protein CUTOFF 8.0 CHAIN "A" WEIGHT "binary" AS G
SEED G NODES [45, 88, 103] STRENGTH 1.0 AS signal
DIFFUSE signal STEPS 100 DT 0.05 MODEL "diffusion" AS output
SPECTRAL G MODES 20 LAPLACIAN "normalized" AS spectrum
COUNTERFACTUAL G REWEIGHT_EDGE 45 88 FACTOR 0.25 COMPARE signal AS weakened_contact
EXPORT spectrum TO "spectrum.json" FORMAT json
'''

EXAMPLE_COUNTERFACTUAL = '''\
LOAD DEMO "synthetic" AS G
SEED G NODE 0 AS seed0
COUNTERFACTUAL G BLOCK_PATH [0, 1, 2, 3, 4] COMPARE seed0 AS blocked
DESCRIBE blocked
'''

EXAMPLES: Dict[str, str] = {
    "basic": EXAMPLE_BASIC,
    "pdb": EXAMPLE_PDB,
    "counterfactual": EXAMPLE_COUNTERFACTUAL,
}


def help_text(topic: Optional[str] = None) -> str:
    topic_key = (topic or "").lower().strip()
    if topic_key in EXAMPLES:
        return EXAMPLES[topic_key]
    if topic_key == "commands":
        return (
            "Commands:\n"
            "  LOAD [PDB|NPZ|CSV|DEMO] <path-or-name> AS <name>\n"
            "  GRAPH <protein> CUTOFF <float> [CHAIN <str>] [WEIGHT <mode>] AS <graph>\n"
            "  SEED <graph> NODE <int> [STRENGTH <float>] AS <signal>\n"
            "  SEED <graph> NODES [1,2,3] [STRENGTH <float>] AS <signal>\n"
            "  DIFFUSE <signal> STEPS <int> DT <float> [MODEL <name>] AS <output>\n"
            "  SPECTRAL <graph> MODES <int> [LAPLACIAN <kind>] AS <spectrum>\n"
            "  COUNTERFACTUAL <graph> REMOVE_NODE <i> AS <report>\n"
            "  COUNTERFACTUAL <graph> REMOVE_EDGE <i> <j> AS <report>\n"
            "  COUNTERFACTUAL <graph> REWEIGHT_EDGE <i> <j> FACTOR <f> AS <report>\n"
            "  EXPORT <name> TO <path> [FORMAT csv|json|npz|markdown]\n"
        )
    return (
        "RINetScript is a compact DSL for residue-interaction network experiments.\n"
        "Type HELP commands for syntax or HELP basic / HELP pdb / HELP counterfactual for examples."
    )


# =============================================================================
# REPL
# =============================================================================


class RINetREPL(cmd.Cmd):
    intro = "RINetScript REPL. Type HELP, :symbols, :examples, :quit."
    prompt = "rinet> "

    def __init__(self, hooks: Optional[RINetExecutionHooks] = None, cwd: Optional[Union[str, os.PathLike[str]]] = None):
        super().__init__()
        self.env = RuntimeEnvironment()
        self.hooks = hooks or RINetExecutionHooks()
        self.cwd = pathlib.Path(cwd or os.getcwd())
        self.multiline_buffer: List[str] = []

    def default(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if line.startswith(":"):
            self._meta_command(line)
            return
        self._run_line(line)

    def do_HELP(self, arg: str) -> None:  # noqa: N802 - cmd uses do_* names
        print(help_text(arg.strip() or None))

    def do_help(self, arg: str) -> None:
        print(help_text(arg.strip() or None))

    def do_exit(self, arg: str) -> bool:
        return True

    def do_quit(self, arg: str) -> bool:
        return True

    def do_EOF(self, arg: str) -> bool:  # noqa: N802
        print()
        return True

    def _run_line(self, line: str) -> None:
        try:
            result = run_script(line, hooks=self.hooks, env=self.env, cwd=self.cwd, analyze=False)
            for message in result.messages:
                print(message)
            last = result.last()
            if last is not None and not isinstance(last, TextValue):
                print(last.summary())
        except RINetLanguageError as exc:
            print(exc.render(line), file=sys.stderr)
        except Exception:
            traceback.print_exc()

    def _meta_command(self, line: str) -> None:
        if line in {":quit", ":exit"}:
            raise SystemExit(0)
        if line == ":symbols":
            for rec in self.env.records():
                print(f"{rec.name}: {rec.value.summary()}")
            return
        if line == ":examples":
            for name in sorted(EXAMPLES):
                print(f"--- {name} ---")
                print(EXAMPLES[name])
            return
        if line.startswith(":format "):
            text = line[len(":format "):]
            try:
                print(format_script(text))
            except RINetLanguageError as exc:
                print(exc.render(text), file=sys.stderr)
            return
        print("Unknown meta-command. Use :symbols, :examples, :format <script>, :quit.")


def repl(hooks: Optional[RINetExecutionHooks] = None, cwd: Optional[Union[str, os.PathLike[str]]] = None) -> None:
    RINetREPL(hooks=hooks, cwd=cwd).cmdloop()


# =============================================================================
# Small command-line entry point for direct module execution
# =============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m allograph.core.rinet_language", description="Run or inspect RINetScript files")
    parser.add_argument("script", nargs="?", help="Script file to run. If omitted, start a REPL.")
    parser.add_argument("--format", action="store_true", help="Print canonical formatting instead of executing.")
    parser.add_argument("--tokens", action="store_true", help="Print tokens instead of executing.")
    parser.add_argument("--no-analyze", action="store_true", help="Skip static semantic analysis.")
    args = parser.parse_args(argv)

    if not args.script:
        repl()
        return 0
    path = pathlib.Path(args.script)
    source = path.read_text(encoding="utf-8")
    try:
        if args.tokens:
            for token in tokenize_script(source):
                print(token.describe())
            return 0
        if args.format:
            print(format_script(source))
            return 0
        result = run_script(source, cwd=path.parent, analyze=not args.no_analyze)
        for message in result.messages:
            print(message)
        if result.last() is not None:
            print(result.last().summary())
        return 0
    except RINetLanguageError as exc:
        print(exc.render(source), file=sys.stderr)
        return 2


__all__ = [
    "TokenKind",
    "Token",
    "Lexer",
    "Parser",
    "Program",
    "Statement",
    "LoadStatement",
    "GraphStatement",
    "SeedStatement",
    "DiffuseStatement",
    "SpectralStatement",
    "CounterfactualStatement",
    "ExportStatement",
    "RuntimeValue",
    "ProteinValue",
    "GraphValue",
    "SignalValue",
    "DiffusionValue",
    "SpectrumValue",
    "CounterfactualValue",
    "RuntimeEnvironment",
    "SemanticAnalyzer",
    "RINetExecutionHooks",
    "RINetInterpreter",
    "ExecutionResult",
    "RINetREPL",
    "parse_script",
    "analyze_script",
    "run_script",
    "format_script",
    "tokenize_script",
    "repl",
    "EXAMPLES",
    "EXAMPLE_BASIC",
    "EXAMPLE_PDB",
    "EXAMPLE_COUNTERFACTUAL",
]


if __name__ == "__main__":
    raise SystemExit(main())
