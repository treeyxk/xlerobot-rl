# Diagnostic Scripts

这里的脚本是为了**临时解决某个具体问题**写的, 不是项目主线代码。

## 当前内容

- `diagnose_robot_motion.py` (2026-05-13)
  - 目的: Day 4 验证 random action 下 17 个 joints 是否都在动
  - 结论: 都在动. **可删除或归档**

- `diagnose_urdf_physics.py` (2026-05-13)
  - 目的: Day 4 验证修改版 URDF 的 joint dynamics 参数
  - 结论: URDF stiffness=0 是预期的, 由 ManiSkill controller 运行时设置
  - **可保留作为 URDF 物理诊断模板**

- `test_urdf_with_drive.py` (2026-05-13)
  - 目的: Day 4 验证加 PD drive 后修改版 URDF 不再下垂
  - 结论: URDF 完全有效
  - **可保留作为 URDF 接入测试模板**

## 维护原则

1. 每个脚本顶部写明 **创建日期 + 目的 + 已解决的问题**
2. 问题解决后, 6 个月还没复用就**删掉**
3. 有复用价值的逻辑应该**重构进 `xlerobot_rl/`**, 不留在 scripts 里
