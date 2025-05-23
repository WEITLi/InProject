#!/usr/bin/env python3
"""测试配置重用功能"""

def test_config_parsing():
    """测试配置解析"""
    from feature_extraction import parse_config_id
    
    configs = [
        "u200_w0-29_msession_s1",
        "u400_w0-29_msession_s1", 
        "uall_w0-74_mweekdaysession_s0"
    ]
    
    print("=== 配置解析测试 ===")
    for config in configs:
        result = parse_config_id(config)
        print(f"{config} -> {result}")

def test_compatibility():
    """测试兼容性检查"""
    from feature_extraction import parse_config_id
    
    print("\n=== 兼容性测试 ===")
    
    # 测试场景：u200_w0-29_msession_s1 是否可以重用 u400_w0-29_msession_s1
    target = parse_config_id("u200_w0-29_msession_s1")
    source = parse_config_id("u400_w0-29_msession_s1")
    
    print(f"目标配置: {target}")
    print(f"源配置: {source}")
    
    # 检查兼容性
    compatible = True
    reasons = []
    
    # 用户数检查
    if target['max_users'] != 'all' and source['max_users'] != 'all':
        if source['max_users'] >= target['max_users']:
            print("✓ 用户数兼容")
        else:
            compatible = False
            reasons.append("用户数不足")
    
    # 周数范围检查
    if (source['start_week'] <= target['start_week'] and 
        source['end_week'] >= target['end_week']):
        print("✓ 周数范围兼容")
    else:
        compatible = False
        reasons.append("周数范围不兼容")
    
    # 模式检查
    if target['modes'] in source['modes']:
        print("✓ 模式兼容")
    else:
        compatible = False
        reasons.append("模式不兼容")
    
    # 子会话检查
    if source['enable_subsession'] == target['enable_subsession']:
        print("✓ 子会话设置兼容")
    else:
        compatible = False
        reasons.append("子会话设置不兼容")
    
    if compatible:
        print("结论: ✓ 完全兼容，可以重用数据")
    else:
        print(f"结论: ✗ 不兼容 - {', '.join(reasons)}")

if __name__ == "__main__":
    test_config_parsing()
    test_compatibility() 