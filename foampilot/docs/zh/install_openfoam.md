# Windows 上的 OpenFOAM – WSL 安装指南

**Windows Subsystem for Linux (WSL)** 允许您在 Windows 10 和 11 上直接运行完整的 Linux 环境。这是目前在 Windows 上可靠运行 OpenFOAM 的推荐方法。

---

## 下载链接

- OpenFOAM 官方网站: https://www.openfoam.com  
- OpenFOAM v13: https://www.openfoam.com/download  
- OpenFOAM-dev: https://www.openfoam.com/download/dev  
- 在 Windows 上运行 OpenFOAM (WSL): https://www.openfoam.com/download/windows  
- 在 macOS 上运行 OpenFOAM: https://www.openfoam.com/download/mac  
- 从源代码编译: https://www.openfoam.com/download/source  
- 软件包仓库: https://dl.openfoam.org  
- 版本历史: https://www.openfoam.com/releases  

---

## 安装 WSL

### 1. 以管理员身份打开 Windows 命令提示符
- 点击 **开始菜单**，输入 `cmd`  
- 右键 **命令提示符** → *以管理员身份运行*  
- 确认所有授权提示  

### 2. 检查 WSL 是否已安装
运行命令：

```bash
wsl -l -v
````

* 如果未安装任何发行版，请继续下一步
* 否则，检查 Ubuntu 是否列出并且版本为 2

### 3. 安装 WSL 和 Ubuntu 22.04

运行：

```bash
wsl --install -d Ubuntu-22.04
```

* 按提示设置 Linux 用户名和密码

### 4. 启动 WSL

* 在 **开始菜单** 中搜索 **Ubuntu** 并打开
* 或在命令提示符中输入：

```bash
wsl
```

---

## 安装 OpenFOAM 和 ParaView

OpenFOAM 和 ParaView 可以通过 `apt` 包管理器安装。以下命令需要超级用户权限 (`sudo`)。

### 1. 添加 OpenFOAM 仓库和公钥

在终端中运行：

```bash
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
```

**注意:**

* 公钥使用 `https://`
* 仓库 URL 使用 `http://`，安全性由 GPG 密钥保证

### 2. 更新软件包列表

```bash
sudo apt update
```

### 3. 安装 OpenFOAM 13

```bash
sudo apt -y install openfoam13
```

OpenFOAM 13 和 ParaView 会安装在 `/opt` 目录下。

---

## 用户配置

要使用 OpenFOAM，请按以下步骤操作：

1. 使用编辑器打开家目录下的 `.bashrc` 文件，例如：

```bash
gedit ~/.bashrc
```

2. 在文件末尾添加以下行，然后保存并关闭：

```bash
. /opt/openfoam13/etc/bashrc
```

3. 打开一个新的终端窗口并测试安装：

```bash
foamRun -help
```

4. 如果显示帮助信息，说明安装和配置已完成。

---

### 注意事项

* 如果 `.bashrc` 中已有类似的旧版本 OpenFOAM 配置，请用 `#` 注释或删除
* 修改 `.bashrc` 后，要立即应用更改，请运行：

```bash
. $HOME/.bashrc
```

---

## 图形库和 LaTeX 依赖

### 安装 Gmsh 和 OpenGL 所需库

```bash
sudo apt install libglu1-mesa libgl1-mesa-glx libxrender1 libxext6
```

### 安装 TexLive（LaTeX）

#### 1. 基础安装

```bash
sudo apt-get install texlive-latex-base
```

#### 2. 推荐字体和扩展字体

为了避免在处理含多字体的 LaTeX 文件时使用 `pdflatex` 出现错误：

```bash
sudo apt-get install texlive-fonts-recommended
sudo apt-get install texlive-fonts-extra
```

#### 3. 额外 LaTeX 软件包

```bash
sudo apt-get install texlive-latex-extra
```

