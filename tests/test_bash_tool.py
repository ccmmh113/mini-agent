"""Test cases for Bash Tool."""

import asyncio
import json
import platform

import pytest

from mini_agent.runtime import RunContext, ToolRuntime
from mini_agent.tools.bash_tool import BackgroundShellManager, BashKillTool, BashOutputTool, BashTool
from mini_agent.tools.security import BashSecurityPolicy, check_command_security


@pytest.mark.asyncio
async def test_foreground_command():
    """Test executing a simple foreground command."""
    print("\n=== Testing Foreground Command ===")

    bash_tool = BashTool()
    result = await bash_tool.execute(command="echo 'Hello from foreground'")

    assert result.success
    assert "Hello from foreground" in result.stdout
    assert result.exit_code == 0
    print(f"Output: {result.content}")


@pytest.mark.asyncio
async def test_foreground_command_with_stderr():
    """Test command that outputs to both stdout and stderr."""
    print("\n=== Testing Stdout/Stderr Separation ===")

    bash_tool = BashTool()
    if platform.system() == "Windows":
        command = "Write-Output 'stdout message'; [Console]::Error.WriteLine('stderr message')"
    else:
        command = "echo 'stdout message' && echo 'stderr message' >&2"
    result = await bash_tool.execute(command=command)

    assert result.success
    assert "stdout message" in result.stdout
    assert "stderr message" in result.stderr
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")


@pytest.mark.asyncio
async def test_command_failure():
    """Test command that fails with non-zero exit code."""
    print("\n=== Testing Command Failure ===")

    bash_tool = BashTool()
    result = await bash_tool.execute(command="ls /nonexistent_directory_12345")

    assert not result.success
    assert result.exit_code != 0
    assert result.error is not None
    print(f"Error: {result.error}")


@pytest.mark.asyncio
async def test_command_timeout():
    """Test command timeout."""
    print("\n=== Testing Command Timeout ===")

    bash_tool = BashTool()
    result = await bash_tool.execute(command="sleep 10", timeout=1)

    assert not result.success
    assert "timed out" in result.error.lower()
    assert result.exit_code == -1
    print(f"Timeout error: {result.error}")


@pytest.mark.asyncio
async def test_background_command():
    """Test running a command in the background."""
    print("\n=== Testing Background Command ===")

    bash_tool = BashTool()
    result = await bash_tool.execute(
        command="for i in 1 2 3; do echo 'Line '$i; sleep 0.5; done", run_in_background=True
    )

    assert result.success
    assert result.bash_id is not None
    assert "Background command started" in result.stdout

    bash_id = result.bash_id
    print(f"Background command started with ID: {bash_id}")

    # Wait a bit for output
    await asyncio.sleep(1)

    # Check output
    bash_output_tool = BashOutputTool()
    output_result = await bash_output_tool.execute(bash_id=bash_id)

    assert output_result.success
    print(f"Output:\n{output_result.content}")

    # Clean up - terminate the background process
    bash_kill_tool = BashKillTool()
    kill_result = await bash_kill_tool.execute(bash_id=bash_id)
    assert kill_result.success
    print("Background process terminated")


@pytest.mark.asyncio
async def test_bash_output_monitoring():
    """Test monitoring background command output."""
    print("\n=== Testing Output Monitoring ===")

    bash_tool = BashTool()

    # Start background command
    result = await bash_tool.execute(
        command="for i in 1 2 3 4 5; do echo 'Line '$i; sleep 0.5; done", run_in_background=True
    )

    assert result.success
    bash_id = result.bash_id
    print(f"Started background command: {bash_id}")

    bash_output_tool = BashOutputTool()

    # Check output multiple times (incremental output)
    for i in range(3):
        await asyncio.sleep(1)
        output_result = await bash_output_tool.execute(bash_id=bash_id)
        assert output_result.success
        print(f"\n--- Check #{i + 1} ---")
        print(f"Output:\n{output_result.content}")

    # Clean up
    bash_kill_tool = BashKillTool()
    await bash_kill_tool.execute(bash_id=bash_id)


@pytest.mark.asyncio
async def test_bash_output_with_filter():
    """Test bash_output with regex filter."""
    print("\n=== Testing Output Filter ===")

    bash_tool = BashTool()

    # Start background command
    result = await bash_tool.execute(
        command="for i in 1 2 3 4 5; do echo 'Line '$i; sleep 0.3; done", run_in_background=True
    )

    assert result.success
    bash_id = result.bash_id

    # Wait for some output
    await asyncio.sleep(2)

    # Get filtered output (only lines with "Line 2" or "Line 4")
    bash_output_tool = BashOutputTool()
    output_result = await bash_output_tool.execute(bash_id=bash_id, filter_str="Line [24]")

    assert output_result.success
    lines = output_result.content
    print(f"Filtered output:\n{output_result.content}")

    # Clean up
    bash_kill_tool = BashKillTool()
    await bash_kill_tool.execute(bash_id=bash_id)


@pytest.mark.asyncio
async def test_bash_kill():
    """Test terminating a background command."""
    print("\n=== Testing Bash Kill ===")

    bash_tool = BashTool()

    # Start a long-running background command
    result = await bash_tool.execute(command="sleep 100", run_in_background=True)

    assert result.success
    bash_id = result.bash_id
    print(f"Started long-running command: {bash_id}")

    # Verify it's running
    await asyncio.sleep(0.5)
    bg_shell = BackgroundShellManager.get(bash_id)
    assert bg_shell is not None
    assert bg_shell.status == "running"

    # Kill it
    bash_kill_tool = BashKillTool()
    kill_result = await bash_kill_tool.execute(bash_id=bash_id)

    assert kill_result.success
    # exit_code -15 means terminated by SIGTERM
    assert kill_result.exit_code == -15 or kill_result.bash_id == bash_id
    print(f"Kill result:\n{kill_result.content}")

    # Verify it's removed from manager
    bg_shell = BackgroundShellManager.get(bash_id)
    assert bg_shell is None


@pytest.mark.asyncio
async def test_bash_kill_nonexistent():
    """Test killing a non-existent bash process."""
    print("\n=== Testing Kill Non-existent Process ===")

    bash_kill_tool = BashKillTool()
    result = await bash_kill_tool.execute(bash_id="nonexistent123")

    assert not result.success
    assert "not found" in result.error.lower()
    print(f"Expected error: {result.error}")


@pytest.mark.asyncio
async def test_bash_output_nonexistent():
    """Test getting output from non-existent bash process."""
    print("\n=== Testing Output From Non-existent Process ===")

    bash_output_tool = BashOutputTool()
    result = await bash_output_tool.execute(bash_id="nonexistent123")

    assert not result.success
    assert "not found" in result.error.lower()
    print(f"Expected error: {result.error}")


@pytest.mark.asyncio
async def test_multiple_background_commands():
    """Test running multiple background commands simultaneously."""
    print("\n=== Testing Multiple Background Commands ===")

    bash_tool = BashTool()

    # Start multiple background commands
    bash_ids = []
    for i in range(3):
        result = await bash_tool.execute(
            command=f"for j in 1 2 3; do echo 'Command {i + 1} Line '$j; sleep 0.5; done", run_in_background=True
        )
        assert result.success
        bash_ids.append(result.bash_id)
        print(f"Started command {i + 1}: {result.bash_id}")

    # Wait and check all commands
    await asyncio.sleep(1)

    bash_output_tool = BashOutputTool()
    for bash_id in bash_ids:
        output_result = await bash_output_tool.execute(bash_id=bash_id)
        assert output_result.success
        print(f"\nOutput for {bash_id}:\n{output_result.content[:100]}...")

    # Clean up all
    bash_kill_tool = BashKillTool()
    for bash_id in bash_ids:
        await bash_kill_tool.execute(bash_id=bash_id)

    print("All background processes cleaned up")


@pytest.mark.asyncio
async def test_timeout_validation():
    """Test timeout parameter validation."""
    print("\n=== Testing Timeout Validation ===")

    bash_tool = BashTool()

    # Test with timeout > 600 (should be capped to 600)
    result = await bash_tool.execute(command="echo 'test'", timeout=1000)
    assert result.success
    print("Timeout > 600 handled correctly")

    # Test with timeout < 1 (should be set to 120)
    result = await bash_tool.execute(command="echo 'test'", timeout=0)
    assert result.success
    print("Timeout < 1 handled correctly")


@pytest.mark.asyncio
async def test_security_blocks_dangerous_command(tmp_path):
    """Test dangerous shell commands are blocked before execution."""
    runtime = ToolRuntime({"bash": BashTool(workspace_dir=str(tmp_path))}, RunContext(workspace_dir=tmp_path))

    result = await runtime.execute("bash", {"command": "rm -rf ./important"})

    assert not result.success
    assert "blocked by security policy" in result.error.lower()


def test_security_parses_quoted_command_head(tmp_path):
    """Test shell-like parsing catches quoted command-name bypasses."""
    decision = check_command_security("r''m -rf ./important", str(tmp_path), BashSecurityPolicy())

    assert not decision.allowed
    assert "recursive_force_delete" in decision.matched_rules


@pytest.mark.asyncio
async def test_security_blocks_absolute_path_outside_workspace(tmp_path):
    """Test commands that reference paths outside the workspace are blocked."""
    outside_file = tmp_path.parent / "outside_workspace.txt"
    runtime = ToolRuntime({"bash": BashTool(workspace_dir=str(tmp_path))}, RunContext(workspace_dir=tmp_path))

    result = await runtime.execute("bash", {"command": f"echo {outside_file}"})

    assert not result.success
    assert "outside workspace" in result.error.lower()


@pytest.mark.asyncio
async def test_security_audit_log_records_allowed_and_completed_events(tmp_path):
    """Test allowed commands write audit events."""
    policy = BashSecurityPolicy(audit_enabled=True)
    bash_tool = BashTool(workspace_dir=str(tmp_path), security_policy=policy)
    runtime = ToolRuntime({"bash": bash_tool}, RunContext(workspace_dir=tmp_path))

    result = await runtime.execute("bash", {"command": "echo audit-ok"})

    assert result.success

    audit_log = tmp_path / ".mini_agent" / "bash_audit.jsonl"
    assert audit_log.exists()

    records = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()]
    decisions = [record["decision"] for record in records]
    assert "allowed" in decisions
    assert "completed" in decisions
    assert records[-1]["success"] is True


@pytest.mark.asyncio
async def test_completed_background_shells_are_cleaned_after_output_is_read():
    """Test completed background shells do not stay in the manager forever."""
    bash_tool = BashTool()
    command = "Write-Output cleanup-done" if platform.system() == "Windows" else "echo cleanup-done"

    result = await bash_tool.execute(command=command, run_in_background=True)
    assert result.success
    assert result.bash_id is not None

    bash_id = result.bash_id
    for _ in range(20):
        shell = BackgroundShellManager.get(bash_id)
        if shell and shell.status != "running":
            break
        await asyncio.sleep(0.1)

    output_tool = BashOutputTool()
    await output_tool.execute(bash_id=bash_id)

    BackgroundShellManager.cleanup_completed(retention_seconds=0)

    assert BackgroundShellManager.get(bash_id) is None
