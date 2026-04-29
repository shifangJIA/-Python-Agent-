"""
Multi-Agent Automated Test Generation System

Features:
- PlannerAgent: analyzes source code and creates a test strategy
- CodeAnalyzerAgent: extracts functions/classes and risk points from Python code
- TestCaseAgent: generates pytest test cases
- EdgeCaseAgent: supplements boundary/exception tests
- CriticAgent: reviews generated tests for missing imports, syntax, and weak coverage
- ExecutorAgent: writes files and runs pytest
- Orchestrator: coordinates the full multi-agent workflow

Usage:
    python multi_agent_test_generator.py --source ./sample_app.py --out ./generated_tests
    python multi_agent_test_generator.py --source ./src --out ./generated_tests --run

Requirements:
    pip install pytest

Optional:
    If you have an LLM API, you can replace RuleBasedLLM.generate() with your API call.
    The current implementation is fully local and rule-based, so it can run immediately.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# =========================
# Shared data structures
# =========================

@dataclasses.dataclass
class FunctionInfo:
    name: str
    qualified_name: str
    args: List[str]
    defaults_count: int
    returns: Optional[str]
    docstring: str
    lineno: int
    end_lineno: int
    is_method: bool = False
    class_name: Optional[str] = None
    decorators: List[str] = dataclasses.field(default_factory=list)
    raises: List[str] = dataclasses.field(default_factory=list)
    branches: int = 0
    calls: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ClassInfo:
    name: str
    docstring: str
    lineno: int
    methods: List[FunctionInfo] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ModuleAnalysis:
    path: str
    module_name: str
    imports: List[str]
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    risk_notes: List[str]


@dataclasses.dataclass
class TestPlan:
    source_path: str
    module_name: str
    goals: List[str]
    targets: List[str]
    risks: List[str]
    strategy: Dict[str, List[str]]


@dataclasses.dataclass
class AgentMessage:
    sender: str
    content: str
    payload: Dict[str, Any] = dataclasses.field(default_factory=dict)


# =========================
# Utility functions
# =========================

def snake_case(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def safe_module_name(path: Path, root: Path) -> str:
    rel = path.resolve().relative_to(root.resolve()) if path.resolve().is_relative_to(root.resolve()) else path.name
    if isinstance(rel, Path):
        parts = list(rel.with_suffix("").parts)
    else:
        parts = [str(rel)]
    return ".".join(p for p in parts if p != "__init__")


def has_package_init(path: Path, root: Path) -> bool:
    try:
        rel_parent = path.parent.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    cur = root
    for part in rel_parent.parts:
        cur = cur / part
        if not (cur / "__init__.py").exists():
            return False
    return True


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def unparse_annotation(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def decorator_name(dec: ast.AST) -> str:
    try:
        return ast.unparse(dec)
    except Exception:
        return "unknown"


def infer_dummy_value(arg_name: str) -> str:
    lower = arg_name.lower()
    if lower in {"n", "num", "count", "index", "idx", "age", "size", "limit", "offset", "x", "y", "a", "b"}:
        return "1"
    if "name" in lower or "text" in lower or "str" in lower or "title" in lower:
        return '"sample"'
    if lower.startswith("is_") or lower.startswith("has_") or lower in {"flag", "enabled", "active"}:
        return "True"
    if "items" in lower or "list" in lower:
        return "[1, 2, 3]"
    if "dict" in lower or "map" in lower or "config" in lower:
        return "{}"
    if "path" in lower or "file" in lower:
        return '"/tmp/sample"'
    return "1"


def import_statement_for(module_name: str, symbol_names: Sequence[str]) -> str:
    names = ", ".join(sorted(set(symbol_names)))
    return f"from {module_name} import {names}"


# =========================
# Local LLM adapter
# =========================

class RuleBasedLLM:
    """
    A swappable LLM facade.

    This local implementation does not call any external API. In production,
    replace generate() with your provider call, but keep the same interface.
    """

    def generate(self, system_prompt: str, user_prompt: str, payload: Dict[str, Any]) -> str:
        intent = payload.get("intent")
        if intent == "summarize_plan":
            targets = payload.get("targets", [])
            risks = payload.get("risks", [])
            return "\n".join([
                "Test goals:",
                *[f"- Exercise {t}" for t in targets],
                "Risk focus:",
                *[f"- {r}" for r in risks],
            ])
        return ""


# =========================
# Base Agent
# =========================

class BaseAgent:
    def __init__(self, name: str, llm: Optional[RuleBasedLLM] = None) -> None:
        self.name = name
        self.llm = llm or RuleBasedLLM()

    def send(self, content: str, **payload: Any) -> AgentMessage:
        return AgentMessage(sender=self.name, content=content, payload=payload)


# =========================
# Agent 1: Code Analyzer
# =========================

class CodeAnalyzerAgent(BaseAgent):
    def __init__(self, llm: Optional[RuleBasedLLM] = None) -> None:
        super().__init__("CodeAnalyzerAgent", llm)

    def analyze_file(self, source_path: Path, project_root: Path) -> ModuleAnalysis:
        code = read_text(source_path)
        tree = ast.parse(code)
        module_name = source_path.stem
        if source_path.parent != project_root and has_package_init(source_path, project_root):
            module_name = safe_module_name(source_path, project_root)

        imports: List[str] = []
        functions: List[FunctionInfo] = []
        classes: List[ClassInfo] = []
        risk_notes: List[str] = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                try:
                    imports.append(ast.unparse(node))
                except Exception:
                    imports.append(type(node).__name__)
            elif isinstance(node, ast.FunctionDef):
                functions.append(self._function_info(node, class_name=None))
            elif isinstance(node, ast.AsyncFunctionDef):
                risk_notes.append(f"Async function {node.name} requires pytest-asyncio or equivalent.")
                functions.append(self._function_info(node, class_name=None))
            elif isinstance(node, ast.ClassDef):
                class_info = ClassInfo(
                    name=node.name,
                    docstring=ast.get_docstring(node) or "",
                    lineno=node.lineno,
                    methods=[],
                )
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_info.methods.append(self._function_info(child, class_name=node.name))
                classes.append(class_info)

        for fn in functions:
            risk_notes.extend(self._risk_notes_for_function(fn))
        for cls in classes:
            for method in cls.methods:
                risk_notes.extend(self._risk_notes_for_function(method))

        return ModuleAnalysis(
            path=str(source_path),
            module_name=module_name,
            imports=imports,
            functions=functions,
            classes=classes,
            risk_notes=sorted(set(risk_notes)),
        )

    def _function_info(self, node: ast.AST, class_name: Optional[str]) -> FunctionInfo:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        args = [a.arg for a in node.args.args]
        if class_name and args and args[0] in {"self", "cls"}:
            display_args = args[1:]
        else:
            display_args = args

        raises: List[str] = []
        calls: List[str] = []
        branches = 0
        for sub in ast.walk(node):
            if isinstance(sub, ast.Raise):
                if sub.exc is None:
                    raises.append("re-raise")
                else:
                    raises.append(unparse_annotation(sub.exc) or "Exception")
            elif isinstance(sub, (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp, ast.Match)):
                branches += 1
            elif isinstance(sub, ast.Call):
                try:
                    calls.append(ast.unparse(sub.func))
                except Exception:
                    pass

        return FunctionInfo(
            name=node.name,
            qualified_name=f"{class_name}.{node.name}" if class_name else node.name,
            args=display_args,
            defaults_count=len(node.args.defaults),
            returns=unparse_annotation(node.returns),
            docstring=ast.get_docstring(node) or "",
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            is_method=class_name is not None,
            class_name=class_name,
            decorators=[decorator_name(d) for d in node.decorator_list],
            raises=raises,
            branches=branches,
            calls=sorted(set(calls)),
        )

    def _risk_notes_for_function(self, fn: FunctionInfo) -> List[str]:
        notes: List[str] = []
        if fn.branches >= 3:
            notes.append(f"{fn.qualified_name} has multiple branches; add branch and edge-case tests.")
        if fn.raises:
            notes.append(f"{fn.qualified_name} raises exceptions; add negative-path tests.")
        if any(c in {"open", "requests.get", "requests.post", "subprocess.run"} or c.startswith("requests.") for c in fn.calls):
            notes.append(f"{fn.qualified_name} touches IO/network/processes; use mocks or tmp_path.")
        if not fn.args:
            notes.append(f"{fn.qualified_name} has no inputs; verify deterministic return or side effects.")
        return notes


# =========================
# Agent 2: Planner
# =========================

class PlannerAgent(BaseAgent):
    def __init__(self, llm: Optional[RuleBasedLLM] = None) -> None:
        super().__init__("PlannerAgent", llm)

    def create_plan(self, analysis: ModuleAnalysis) -> TestPlan:
        targets: List[str] = [fn.qualified_name for fn in analysis.functions]
        for cls in analysis.classes:
            targets.append(cls.name)
            targets.extend(m.qualified_name for m in cls.methods if not m.name.startswith("__"))

        goals = [
            "Generate executable pytest tests",
            "Cover normal path, boundary values, and exception paths",
            "Avoid external side effects by using mocks or temporary paths when needed",
            "Produce maintainable test names and clear assertions",
        ]

        strategy: Dict[str, List[str]] = {}
        for target in targets:
            strategy[target] = ["happy_path", "edge_case", "negative_path"]

        return TestPlan(
            source_path=analysis.path,
            module_name=analysis.module_name,
            goals=goals,
            targets=targets,
            risks=analysis.risk_notes,
            strategy=strategy,
        )


# =========================
# Agent 3: Test Case Generator
# =========================

class TestCaseAgent(BaseAgent):
    def __init__(self, llm: Optional[RuleBasedLLM] = None) -> None:
        super().__init__("TestCaseAgent", llm)

    def generate_tests(self, analysis: ModuleAnalysis, plan: TestPlan) -> str:
        import_symbols: List[str] = []
        import_symbols.extend(fn.name for fn in analysis.functions if not fn.name.startswith("_"))
        import_symbols.extend(cls.name for cls in analysis.classes if not cls.name.startswith("_"))

        lines: List[str] = [
            "# Auto-generated by multi_agent_test_generator.py",
            "# Review and adjust assertions for domain-specific behavior.",
            "import pytest",
            "",
        ]
        if import_symbols:
            lines.append(import_statement_for(analysis.module_name, import_symbols))
            lines.append("")

        for fn in analysis.functions:
            if fn.name.startswith("_"):
                continue
            lines.append(self._test_for_function(fn))
            lines.append("")

        for cls in analysis.classes:
            if cls.name.startswith("_"):
                continue
            lines.append(self._test_for_class_instantiation(cls))
            lines.append("")
            for method in cls.methods:
                if method.name.startswith("_"):
                    continue
                lines.append(self._test_for_method(cls, method))
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _call_args(self, fn: FunctionInfo) -> str:
        return ", ".join(infer_dummy_value(a) for a in fn.args)

    def _test_for_function(self, fn: FunctionInfo) -> str:
        args = self._call_args(fn)
        test_name = f"test_{snake_case(fn.name)}_happy_path"
        body = [f"def {test_name}():"]
        if fn.raises and not fn.args:
            body.append(f"    with pytest.raises(Exception):")
            body.append(f"        {fn.name}({args})")
        else:
            body.append(f"    result = {fn.name}({args})")
            body.append("    assert result is not None")
        return "\n".join(body)

    def _test_for_class_instantiation(self, cls: ClassInfo) -> str:
        init_method = next((m for m in cls.methods if m.name == "__init__"), None)
        args = self._call_args(init_method) if init_method else ""
        return "\n".join([
            f"def test_{snake_case(cls.name)}_can_be_instantiated():",
            f"    instance = {cls.name}({args})",
            f"    assert isinstance(instance, {cls.name})",
        ])

    def _test_for_method(self, cls: ClassInfo, method: FunctionInfo) -> str:
        init_method = next((m for m in cls.methods if m.name == "__init__"), None)
        init_args = self._call_args(init_method) if init_method else ""
        method_args = self._call_args(method)
        return "\n".join([
            f"def test_{snake_case(cls.name)}_{snake_case(method.name)}_happy_path():",
            f"    instance = {cls.name}({init_args})",
            f"    result = instance.{method.name}({method_args})",
            "    assert result is not None",
        ])


# =========================
# Agent 4: Edge Case Generator
# =========================

class EdgeCaseAgent(BaseAgent):
    def __init__(self, llm: Optional[RuleBasedLLM] = None) -> None:
        super().__init__("EdgeCaseAgent", llm)

    def generate_edge_tests(self, analysis: ModuleAnalysis) -> str:
        chunks: List[str] = []
        for fn in analysis.functions:
            if fn.name.startswith("_"):
                continue
            chunks.append(self._edge_tests_for_function(fn))
        for cls in analysis.classes:
            if cls.name.startswith("_"):
                continue
            for method in cls.methods:
                if not method.name.startswith("_"):
                    chunks.append(self._edge_tests_for_method(cls, method))
        return "\n\n".join(c for c in chunks if c).strip()

    def _edge_tests_for_function(self, fn: FunctionInfo) -> str:
        tests: List[str] = []
        if fn.args:
            none_args = ", ".join("None" for _ in fn.args)
            tests.append("\n".join([
                f"def test_{snake_case(fn.name)}_rejects_none_inputs():",
                "    with pytest.raises(Exception):",
                f"        {fn.name}({none_args})",
            ]))
        if fn.raises:
            args = ", ".join(infer_dummy_value(a) for a in fn.args)
            tests.append("\n".join([
                f"def test_{snake_case(fn.name)}_documents_exception_behavior():",
                "    try:",
                f"        {fn.name}({args})",
                "    except Exception as exc:",
                "        assert isinstance(exc, Exception)",
            ]))
        return "\n\n".join(tests)

    def _edge_tests_for_method(self, cls: ClassInfo, method: FunctionInfo) -> str:
        if not method.args:
            return ""
        init_method = next((m for m in cls.methods if m.name == "__init__"), None)
        init_args = ", ".join(infer_dummy_value(a) for a in init_method.args) if init_method else ""
        none_args = ", ".join("None" for _ in method.args)
        return "\n".join([
            f"def test_{snake_case(cls.name)}_{snake_case(method.name)}_rejects_none_inputs():",
            f"    instance = {cls.name}({init_args})",
            "    with pytest.raises(Exception):",
            f"        instance.{method.name}({none_args})",
        ])


# =========================
# Agent 5: Critic / Reviewer
# =========================

class CriticAgent(BaseAgent):
    def __init__(self, llm: Optional[RuleBasedLLM] = None) -> None:
        super().__init__("CriticAgent", llm)

    def review_and_fix(self, test_code: str) -> Tuple[str, List[str]]:
        notes: List[str] = []
        fixed = test_code

        if "pytest.raises" in fixed and "import pytest" not in fixed:
            fixed = "import pytest\n" + fixed
            notes.append("Added missing pytest import.")

        fixed = re.sub(r"\n{3,}", "\n\n", fixed)

        try:
            ast.parse(fixed)
        except SyntaxError as exc:
            notes.append(f"Syntax warning: {exc}. Manual review is required.")

        if "assert result is not None" in fixed:
            notes.append("Some assertions are generic. Replace them with exact expected values after reviewing business logic.")

        return fixed, notes


# =========================
# Agent 6: Executor
# =========================

class ExecutorAgent(BaseAgent):
    def __init__(self, llm: Optional[RuleBasedLLM] = None) -> None:
        super().__init__("ExecutorAgent", llm)

    def write_tests(self, out_dir: Path, source_file: Path, test_code: str) -> Path:
        test_name = f"test_{source_file.stem}.py"
        test_path = out_dir / test_name
        write_text(test_path, test_code)
        return test_path

    def run_pytest(self, test_dir: Path, project_root: Path) -> Tuple[int, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_dir), "-q"],
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return proc.returncode, proc.stdout + proc.stderr


# =========================
# Orchestrator
# =========================

class MultiAgentTestOrchestrator:
    def __init__(self) -> None:
        llm = RuleBasedLLM()
        self.analyzer = CodeAnalyzerAgent(llm)
        self.planner = PlannerAgent(llm)
        self.generator = TestCaseAgent(llm)
        self.edge_generator = EdgeCaseAgent(llm)
        self.critic = CriticAgent(llm)
        self.executor = ExecutorAgent(llm)
        self.messages: List[AgentMessage] = []

    def process_file(self, source_file: Path, project_root: Path, out_dir: Path) -> Dict[str, Any]:
        analysis = self.analyzer.analyze_file(source_file, project_root)
        self.messages.append(self.analyzer.send("Analyzed source file", analysis=dataclasses.asdict(analysis)))

        plan = self.planner.create_plan(analysis)
        self.messages.append(self.planner.send("Created test plan", plan=dataclasses.asdict(plan)))

        base_tests = self.generator.generate_tests(analysis, plan)
        edge_tests = self.edge_generator.generate_edge_tests(analysis)
        combined = base_tests.rstrip() + "\n\n" + edge_tests.rstrip() + "\n" if edge_tests else base_tests
        self.messages.append(self.generator.send("Generated base tests"))
        self.messages.append(self.edge_generator.send("Generated edge-case tests"))

        fixed_tests, review_notes = self.critic.review_and_fix(combined)
        self.messages.append(self.critic.send("Reviewed generated tests", notes=review_notes))

        test_path = self.executor.write_tests(out_dir, source_file, fixed_tests)
        self.messages.append(self.executor.send("Wrote test file", test_path=str(test_path)))

        return {
            "source": str(source_file),
            "test_path": str(test_path),
            "analysis": dataclasses.asdict(analysis),
            "plan": dataclasses.asdict(plan),
            "review_notes": review_notes,
        }

    def run(self, source: Path, out_dir: Path, run_tests: bool = False) -> Dict[str, Any]:
        source = source.resolve()
        out_dir = out_dir.resolve()
        project_root = source if source.is_dir() else source.parent

        py_files = self._collect_python_files(source)
        results: List[Dict[str, Any]] = []
        for py_file in py_files:
            if py_file.name.startswith("test_") or py_file.name == "__init__.py":
                continue
            results.append(self.process_file(py_file, project_root, out_dir))

        execution: Optional[Dict[str, Any]] = None
        if run_tests:
            code, output = self.executor.run_pytest(out_dir, project_root)
            execution = {"exit_code": code, "output": output}
            self.messages.append(self.executor.send("Executed pytest", exit_code=code, output=output))

        report = {
            "source": str(source),
            "out_dir": str(out_dir),
            "files_processed": len(results),
            "results": results,
            "execution": execution,
            "messages": [dataclasses.asdict(m) for m in self.messages],
        }
        write_text(out_dir / "agent_report.json", json.dumps(report, indent=2, ensure_ascii=False))
        return report

    def _collect_python_files(self, source: Path) -> List[Path]:
        if source.is_file():
            if source.suffix != ".py":
                raise ValueError("source file must be a .py file")
            return [source]
        if source.is_dir():
            excluded = {".venv", "venv", "__pycache__", ".git", "build", "dist"}
            return [
                p for p in source.rglob("*.py")
                if not any(part in excluded for part in p.parts)
            ]
        raise FileNotFoundError(f"source does not exist: {source}")


# =========================
# Demo project generator
# =========================

def create_demo_file(path: Path) -> None:
    demo_code = '''
class Calculator:
    def __init__(self, precision=2):
        self.precision = precision

    def divide(self, a, b):
        if b == 0:
            raise ValueError("b cannot be zero")
        return round(a / b, self.precision)


def add(a, b):
    return a + b


def normalize_name(name):
    if not name:
        raise ValueError("name is required")
    return name.strip().title()
'''.lstrip()
    write_text(path, demo_code)


# =========================
# CLI
# =========================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-agent automated pytest generator")
    parser.add_argument("--source", type=str, help="Python file or directory to analyze")
    parser.add_argument("--out", type=str, default="generated_tests", help="Output directory for generated tests")
    parser.add_argument("--run", action="store_true", help="Run pytest after generating tests")
    parser.add_argument("--demo", action="store_true", help="Create and test a demo source file")
    args = parser.parse_args(argv)

    if args.demo:
        demo_dir = Path("demo_project").resolve()
        source = demo_dir / "sample_app.py"
        create_demo_file(source)
        out = demo_dir / "generated_tests"
        orchestrator = MultiAgentTestOrchestrator()
        report = orchestrator.run(source, out, run_tests=args.run)
    else:
        if not args.source:
            parser.error("--source is required unless --demo is used")
        orchestrator = MultiAgentTestOrchestrator()
        report = orchestrator.run(Path(args.source), Path(args.out), run_tests=args.run)

    print(json.dumps({
        "files_processed": report["files_processed"],
        "out_dir": report["out_dir"],
        "execution": report["execution"],
    }, indent=2, ensure_ascii=False))
    print(f"\nFull report: {Path(report['out_dir']) / 'agent_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# -Python-Agent-
