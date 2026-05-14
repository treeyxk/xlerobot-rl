# Diagnostic Scripts

这里的脚本是为了**临时解决某个具体问题**写的, 不是项目主线代码。

## 当前内容

### `load_my_urdf.py` (2026-05-13)
- **目的**: Day 4 验证修改版 URDF 能被 SAPIEN URDF loader 直接加载 (绕开 ManiSkill agent 系统)
- **结论**: ✅ 加载成功, URDF 结构有效
- **保留价值**: 中. 未来 Week 2 接入修改版 URDF 到 ManiSkill 时可参考

### `diagnose_robot_motion.py` (2026-05-13)
- **目的**: Day 4 验证 random action 下 17 个 active joints 是否都在动
- **结论**: ✅ 都在动, "静止"是视觉错觉 (单只手动作幅度小)
- **保留价值**: 低. **3 个月后可删**

### `diagnose_urdf_physics.py` (2026-05-13)
- **目的**: 诊断修改版 URDF 的 joint dynamics 参数和 mass/inertia 分布
- **结论**: URDF stiffness=0 是预期 (由 ManiSkill controller 运行时设置). 总质量 13.79 kg, CoM 对称
- **保留价值**: 高. 保留作为 URDF 物理诊断模板

### `test_urdf_with_drive.py` (2026-05-13)
- **目的**: 验证给 URDF 加 PD drive 后机器人是否稳定 (不下垂)
- **结论**: ✅ 加 PD drive 后完美稳定, URDF 本身无问题
- **保留价值**: 高. 保留作为 URDF 接入 ManiSkill 前的最后验证模板

## 维护原则

1. 每个脚本顶部写明 **创建日期 + 目的 + 已解决的问题**
2. 问题解决后, 6 个月还没复用就**删掉**
3. 有复用价值的逻辑应该**重构进 `xlerobot_rl/`**, 不留在 scripts 里
4. 不要为了"以防万一"保留, scripts 增长是不可逆的
