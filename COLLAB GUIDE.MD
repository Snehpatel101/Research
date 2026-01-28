
  Setup Instructions

  1. Open VS Code → Ctrl+Shift+X → search "Google Colab" → Install
  2. It auto-installs the Jupyter extension dependency
  3. Open or create a .ipynb notebook
  4. Click Select Kernel → Colab → New Colab Server
  5. Sign in with your Google account
  6. You now have Colab's GPU/TPU runtime running your local notebooks

  What This Means for You

  You can open your cloned Research repo in VS Code, create a notebook right there, and
  run it on Colab's runtime — all without the browser. Combined with the Claude Code
  extension, you'd have:

  - Full IDE editing of your src/ code and notebooks
  - Colab GPU/TPU running your cells
  - Me (Claude Code) with full filesystem + terminal access

  Limitations to Know

  - No Google Drive mounting (drive.mount() doesn't work)
  - No userdata.get() for Colab Secrets
  - Free tier disconnects after ~90 min idle

  This is the best workflow for your use case. Want me to help you build the notebook once
   you're set up in VS Code?

  Sources:
  - https://marketplace.visualstudio.com/items?itemName=Google.colab
  - https://developers.googleblog.com/google-colab-is-coming-to-vs-code/
  - https://github.com/googlecolab/colab-vscode
  - https://www.infoq.com/news/2025/11/visual-studio-code-colab-bridge/
