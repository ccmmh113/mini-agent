# Model Config Templates

This directory contains safe example configs for real-model evals.

Copy an example file to a local `.yaml` file before running real evals:

```powershell
Copy-Item configs\gpt.yaml.example configs\gpt.yaml
Copy-Item configs\deepseek.yaml.example configs\deepseek.yaml
Copy-Item configs\claude.yaml.example configs\claude.yaml
```

Then replace the example API key locally. Files matching `configs/*.yaml` are ignored by Git so secrets are not committed.
