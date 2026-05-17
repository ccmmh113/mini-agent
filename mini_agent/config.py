"""Configuration management module

Provides unified configuration loading and management functionality
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .context_budget import PromptLayerBudgets
from .schema import TokenPricing


class RetryConfig(BaseModel):
    """Retry configuration"""

    enabled: bool = True
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


class LLMConfig(BaseModel):
    """LLM configuration"""

    api_key: str
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    provider: str = "openai"  # "anthropic" or "openai"
    openai_prompt_cache_key: str | None = None
    openai_prompt_cache_retention: str | None = None
    disable_thinking: bool = False
    enable_reasoning_split: bool = False
    preserve_thinking: bool = False
    show_thinking: bool = False
    log_thinking: bool = False
    token_pricing: TokenPricing = Field(default_factory=TokenPricing)
    retry: RetryConfig = Field(default_factory=RetryConfig)


class ContextLayerBudgetConfig(BaseModel):
    """Per-layer prompt budget configuration."""

    core: int = 2500
    skills: int = 1200
    memory: int = 1200
    project_rules: int = 1800
    current_task_context: int = 1000
    harness_summary: int = 1800
    dynamic_context: int = 300

    def to_prompt_layer_budgets(self) -> PromptLayerBudgets:
        return PromptLayerBudgets(
            core=self.core,
            skills=self.skills,
            memory=self.memory,
            project_rules=self.project_rules,
            current_task_context=self.current_task_context,
            harness_summary=self.harness_summary,
            dynamic_context=self.dynamic_context,
        )


class AgentConfig(BaseModel):
    """Agent configuration"""

    max_steps: int = 50
    workspace_dir: str = "./workspace"
    system_prompt_path: str = "system_prompt.md"
    token_limit: int = 10000
    request_context_limit: int = 12
    context_layer_budgets: ContextLayerBudgetConfig = Field(default_factory=ContextLayerBudgetConfig)


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) timeout configuration"""

    connect_timeout: float = 10.0  # Connection timeout (seconds)
    execute_timeout: float = 60.0  # Tool execution timeout (seconds)
    sse_read_timeout: float = 120.0  # SSE read timeout (seconds)


class ToolsConfig(BaseModel):
    """Tools configuration"""

    # Basic tools (file operations, bash)
    enable_file_tools: bool = True
    enable_bash: bool = True
    enable_note: bool = True
    enable_task_memory: bool = True
    enable_bash_security: bool = True
    enable_bash_confirmation: bool = True
    bash_allowed_commands: list[str] = Field(default_factory=list)
    bash_blocked_commands: list[str] = Field(default_factory=list)
    bash_allow_outside_workspace: bool = False
    bash_audit_enabled: bool = True
    bash_audit_log_path: str | None = None

    # Skills are optional and can be heavy when bundled with example resources.
    enable_skills: bool = False
    skills_dir: str = "./skills"

    # MCP tools
    enable_mcp: bool = False
    mcp_config_path: str = "mcp.json"
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    # Delegated child agents
    enable_subagent: bool = False


class SubagentConfig(BaseModel):
    """Configuration for isolated delegated child agents."""

    max_steps: int = 12
    token_limit: int = 6000
    request_context_limit: int = 8
    allowed_tools: list[str] = Field(default_factory=lambda: ["read_file", "bash", "recall_notes"])
    allow_nested_subagent: bool = False


class Config(BaseModel):
    """Main configuration class"""

    llm: LLMConfig
    agent: AgentConfig
    tools: ToolsConfig
    subagent: SubagentConfig = Field(default_factory=SubagentConfig)

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from the default search path."""
        config_path = cls.get_default_config_path()
        if not config_path.exists():
            raise FileNotFoundError(
                "Configuration file not found. Create config.yaml in mini_agent/config/ "
                "or ~/.mini-agent/config/."
            )
        return cls.from_yaml(config_path)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """Load configuration from YAML file

        Args:
            config_path: Configuration file path

        Returns:
            Config instance

        Raises:
            FileNotFoundError: Configuration file does not exist
            ValueError: Invalid configuration format or missing required fields
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError("Configuration file is empty")

        # Parse LLM configuration
        if "api_key" not in data:
            raise ValueError("Configuration file missing required field: api_key")

        if not data["api_key"] or data["api_key"] == "YOUR_API_KEY_HERE":
            raise ValueError("Please configure a valid API Key")

        # Parse retry configuration
        retry_data = data.get("retry", {})
        retry_config = RetryConfig(
            enabled=retry_data.get("enabled", True),
            max_retries=retry_data.get("max_retries", 3),
            initial_delay=retry_data.get("initial_delay", 1.0),
            max_delay=retry_data.get("max_delay", 60.0),
            exponential_base=retry_data.get("exponential_base", 2.0),
        )

        pricing_data = data.get("token_pricing", {}) or {}
        token_pricing = TokenPricing(
            input_per_1m=pricing_data.get("input_per_1m", 0.0),
            output_per_1m=pricing_data.get("output_per_1m", 0.0),
            cache_read_per_1m=pricing_data.get("cache_read_per_1m", 0.0),
            cache_write_per_1m=pricing_data.get("cache_write_per_1m", 0.0),
            currency=pricing_data.get("currency", "USD"),
        )

        llm_config = LLMConfig(
            api_key=data["api_key"],
            api_base=data.get("api_base", "https://api.openai.com/v1"),
            model=data.get("model", "gpt-4o-mini"),
            provider=data.get("provider", "openai"),
            openai_prompt_cache_key=data.get("openai_prompt_cache_key"),
            openai_prompt_cache_retention=data.get("openai_prompt_cache_retention"),
            disable_thinking=data.get("disable_thinking", False),
            enable_reasoning_split=data.get("enable_reasoning_split", False),
            preserve_thinking=data.get("preserve_thinking", False),
            show_thinking=data.get("show_thinking", False),
            log_thinking=data.get("log_thinking", False),
            token_pricing=token_pricing,
            retry=retry_config,
        )

        # Parse Agent configuration
        context_layer_data = data.get("context_layer_budgets", {})
        agent_config = AgentConfig(
            max_steps=data.get("max_steps", 50),
            workspace_dir=data.get("workspace_dir", "./workspace"),
            system_prompt_path=data.get("system_prompt_path", "system_prompt.md"),
            token_limit=data.get("token_limit", 10000),
            request_context_limit=data.get("request_context_limit", 12),
            context_layer_budgets=ContextLayerBudgetConfig(
                core=context_layer_data.get("core", 2500),
                skills=context_layer_data.get("skills", 1200),
                memory=context_layer_data.get("memory", 1200),
                project_rules=context_layer_data.get("project_rules", 1800),
                current_task_context=context_layer_data.get("current_task_context", 1000),
                harness_summary=context_layer_data.get("harness_summary", 1800),
                dynamic_context=context_layer_data.get("dynamic_context", 300),
            ),
        )

        # Parse tools configuration
        tools_data = data.get("tools", {})

        # Parse MCP configuration
        mcp_data = tools_data.get("mcp", {})
        mcp_config = MCPConfig(
            connect_timeout=mcp_data.get("connect_timeout", 10.0),
            execute_timeout=mcp_data.get("execute_timeout", 60.0),
            sse_read_timeout=mcp_data.get("sse_read_timeout", 120.0),
        )

        tools_config = ToolsConfig(
            enable_file_tools=tools_data.get("enable_file_tools", True),
            enable_bash=tools_data.get("enable_bash", True),
            enable_note=tools_data.get("enable_note", True),
            enable_task_memory=tools_data.get("enable_task_memory", True),
            enable_bash_security=tools_data.get("enable_bash_security", True),
            enable_bash_confirmation=tools_data.get("enable_bash_confirmation", True),
            bash_allowed_commands=tools_data.get("bash_allowed_commands", []),
            bash_blocked_commands=tools_data.get("bash_blocked_commands", []),
            bash_allow_outside_workspace=tools_data.get("bash_allow_outside_workspace", False),
            bash_audit_enabled=tools_data.get("bash_audit_enabled", True),
            bash_audit_log_path=tools_data.get("bash_audit_log_path"),
            enable_skills=tools_data.get("enable_skills", False),
            skills_dir=tools_data.get("skills_dir", "./skills"),
            enable_mcp=tools_data.get("enable_mcp", False),
            mcp_config_path=tools_data.get("mcp_config_path", "mcp.json"),
            mcp=mcp_config,
            enable_subagent=tools_data.get("enable_subagent", False),
        )

        subagent_data = data.get("subagent", {}) or {}
        subagent_config = SubagentConfig(
            max_steps=subagent_data.get("max_steps", 12),
            token_limit=subagent_data.get("token_limit", 6000),
            request_context_limit=subagent_data.get("request_context_limit", 8),
            allowed_tools=subagent_data.get("allowed_tools", ["read_file", "bash", "recall_notes"]),
            allow_nested_subagent=subagent_data.get("allow_nested_subagent", False),
        )

        return cls(
            llm=llm_config,
            agent=agent_config,
            tools=tools_config,
            subagent=subagent_config,
        )

    @staticmethod
    def get_package_dir() -> Path:
        """Get the package installation directory

        Returns:
            Path to the mini_agent package directory
        """
        # Get the directory where this config.py file is located
        return Path(__file__).parent

    @classmethod
    def find_config_file(cls, filename: str) -> Path | None:
        """Find configuration file with priority order

        Search for config file in the following order of priority:
        1) mini_agent/config/{filename} in current directory (development mode)
        2) ~/.mini-agent/config/{filename} in user home directory
        3) {package}/mini_agent/config/{filename} in package installation directory

        Args:
            filename: Configuration file name (e.g., "config.yaml", "mcp.json", "system_prompt.md")

        Returns:
            Path to found config file, or None if not found
        """
        # Priority 1: Development mode - current directory's config/ subdirectory
        dev_config = Path.cwd() / "mini_agent" / "config" / filename
        if dev_config.exists():
            return dev_config

        # Priority 2: User config directory
        user_config = Path.home() / ".mini-agent" / "config" / filename
        if user_config.exists():
            return user_config

        # Priority 3: Package installation directory's config/ subdirectory
        package_config = cls.get_package_dir() / "config" / filename
        if package_config.exists():
            return package_config

        return None

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the default config file path with priority search

        Returns:
            Path to config.yaml (prioritizes: dev config/ > user config/ > package config/)
        """
        config_path = cls.find_config_file("config.yaml")
        if config_path:
            return config_path

        # Fallback to package config directory for error message purposes
        return cls.get_package_dir() / "config" / "config.yaml"
