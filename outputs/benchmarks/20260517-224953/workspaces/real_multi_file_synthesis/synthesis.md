# Synthesis

基于 architecture.md 和 risks.md 的综合分析：

- **harness**：Mini Agent 的核心组件，负责 prompt、工具、状态和执行边界的统一管理。
- **checkpoint**：用于保存可恢复的上下文，确保执行过程可追溯和恢复。
- **Bash policy**：在执行 shell 命令前进行安全拦截和确认，降低风险。
