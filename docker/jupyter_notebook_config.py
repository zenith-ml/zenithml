# flake8: noqa
c.ServerApp.token = "p"
c.ServerApp.password = ""
c.ServerApp.open_browser = False
c.ServerApp.port = 7777
c.ServerApp.allow_origin_pat = "(^https://7777-dot-[0-9]+-dot-devshell\.appspot\.com$)|(^https://colab\.research\.google\.com$)|((https?://)?[0-9a-z]+-dot-datalab-vm[\-0-9a-z]*.googleusercontent.com)|((https?://)?[0-9a-z]+-dot-[\-0-9a-z]*.notebooks.googleusercontent.com)"
c.ServerApp.allow_remote_access = True
c.ServerApp.disable_check_xsrf = False
c.ServerApp.notebook_dir = "/home"