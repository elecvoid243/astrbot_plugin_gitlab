# 在插件数据目录持久保存项目源码和索引

每个绑定项目的只读 Git 副本和源码索引都保存在 AstrBot 主机的 `data/plugin_data/astrbot_plugin_gitlab/` 下，按 GitLab 实例和项目隔离。这样机器人可以在本地检索代码，按提交核实诊断和评审意见；部署方接受私有仓库源码在该主机持久保存。插件通过 `StarTools.get_data_dir()` 定位自己的数据目录。
