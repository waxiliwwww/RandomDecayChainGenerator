import random
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union

class Particle:
    def __init__(self, name: str, properties: Dict):
        """
        粒子类，包含所有物理属性
        """
        self.name = name
        # 基础量子数
        self.charge = properties.get('charge', 0)
        self.strangeness = properties.get('strangeness', 0)
        self.baryon_number = properties.get('baryon_number', 0)
      
        # 高级量子数
        self.quantum_C = properties.get('quantum_C', None)  # 电荷共轭宇称
        self.quantum_G = properties.get('quantum_G', None)  # G宇称
        self.quantum_I = properties.get('quantum_I', None)  # 同位旋
        self.quantum_J = properties.get('quantum_J', None)  # 总角动量
        self.quantum_P = properties.get('quantum_P', None)  # 空间宇称
      
        # 质量与寿命
        self.mass = properties.get('mass', 0)
        self.mass_err = properties.get('mass_err', 0)
        self.width = properties.get('width', 0)
        self.width_err = properties.get('width_err', 0)
        self.lifetime = properties.get('lifetime', float('inf'))
        self.lifetime_err = properties.get('lifetime_err', 0)
      
        # 粒子类型标志
        self.is_baryon = properties.get('is_baryon', False)
        self.is_boson = properties.get('is_boson', False)
        self.is_lepton = properties.get('is_lepton', False)
        self.is_meson = properties.get('is_meson', False)
        self.is_quark = properties.get('is_quark', False)
      
        # 衰变特性
        self.branching_fractions = properties.get('branching_fractions', [])
        self.exclusive_branching_fractions = properties.get('exclusive_branching_fractions', [])
        self.inclusive_branching_fractions = properties.get('inclusive_branching_fractions', [])
      
        # 数据库标记
        self.has_lifetime_entry = properties.get('has_lifetime_entry', False)
        self.has_mass_entry = properties.get('has_mass_entry', False)
        self.has_width_entry = properties.get('has_width_entry', False)
      
        # 计算稳定性
        self.is_stable = properties.get('is_stable', True)
      
        # 中间态标记（用于衰变链）
        self.is_intermediate = False

class ConservationChecker:
    def __init__(self, rules: Dict[str, bool]):
        """
        守恒规则检查器
        :param rules: 守恒规则字典，例如 {'charge': True, 'strangeness': True}
        """
        self.rules = rules
  
    def check(self, parents: List[Particle], children: List[Particle], ecms: float = None) -> bool:
        """检查衰变产物是否满足所有启用的守恒定律"""
        # 计算初始总量子数
        init_values = self._calculate_total_properties(parents)
        if len(parents) > 1:
            init_values['mass'] = ecms
      
        # 计算产物总量子数
        final_values = self._calculate_total_properties(children)
      
        # 检查物理定律是否满足
        for quantum in final_values:
            if quantum == 'mass':
                if final_values[quantum] >= init_values[quantum]:
                    return False
            # 对于其他量，如果有规则且初态和末态不相等，则返回False
            elif self.rules.get(quantum, False) and init_values[quantum] != final_values[quantum]:
                return False
      
        return True

    def _calculate_total_properties(self, particles: List[Particle]) -> Dict:
        """计算量子数总和"""
        return {
            'charge': sum(p.charge for p in particles),
            'mass': sum(p.mass for p in particles),
            'strangeness': sum(p.strangeness for p in particles),
            'baryon_number': sum(p.baryon_number for p in particles),
            'lepton_number': sum(1 for p in particles if p.is_lepton and not p.is_quark),
        }

class DecayChainGenerator:
    _file_cache = None

    def __init__(self, particle_db: Dict[str, Particle], conservation_rules: Dict[str, bool], ecms: float = None, **kwargs):
        self.db = particle_db
        self.conservation = ConservationChecker(conservation_rules)
        self.ecms = ecms
        self.max_depth = kwargs.get('max_depth', 10)  # 最大递归深度
        self.max_final_particles = kwargs.get('max_final_particles', 8)  # 最大末态粒子数
        self.min_final_particles = 2   # 最小末态粒子数
        self.max_attempts = kwargs.get('max_attempts', 1000)  # 最大尝试次数
  
    def generate(self, initial_particles: List[str]) -> str:
        """
        生成完整的衰变链
        :param initial_particles: 初始粒子名称列表
        :return: 完整衰变链字符串
        """
        # 转换粒子名称到对象
        initial_state = [self.db[name] for name in initial_particles]
      
        while True:
            # 生成衰变链
            decay_chain = {
                'initial_state': [p.name for p in initial_state],
                'full_chain': [],
                'final_state': [],
            }

            # 随机生成末态粒子组合
            final_state = self._generate_final_state(initial_state)

            # 递归展开中间态粒子
            expanded_final_state = self._expand_particles(final_state)

            # 记录完整衰变链
            decay_chain['full_chain'] = expanded_final_state
            decay_chain['final_state'] = self._extract_final_particles(expanded_final_state)
            
            processed_decay_chain = self.process_decay_chain(decay_chain)
            if self._file_cache is not None:
                if processed_decay_chain in self._file_cache:
                    continue  # 如果衰变链已在缓存中，继续生成新的衰变链
                self._file_cache.append(processed_decay_chain)
            else:
                self._file_cache = [processed_decay_chain]
            
            decay_chain_str = self.format_decay_chain(decay_chain)
            return decay_chain_str  # 返回不在缓存中的衰变链
  
    def _generate_final_state(self, initial_state: List[Particle]) -> List[Particle]:
        """
        随机生成满足守恒定律的末态粒子组合
        """
        # 获取所有可能的粒子（包括中间态）
        all_particles = list(self.db.values())
      
        for _ in range(self.max_attempts):
            # 随机决定末态粒子数量
            num_particles = random.randint(self.min_final_particles, self.max_final_particles)
          
            # 随机选择粒子组合
            final_state = random.sample(all_particles, num_particles)
          
            # 检查量子数守恒
            if self.conservation.check(initial_state, final_state, self.ecms):
                return final_state
      
        # 如果多次尝试失败，返回默认组合
        return [Particle("UNKNOWN", {'charge': 0, 'mass': 0, 'is_stable': True}), Particle("UNKNOWN", {'charge': 0, 'mass': 0, 'is_stable': True})]
  
    def _expand_particles(self, particles: List[Particle], depth: int = 0) -> List[Union[Dict, Particle]]:
        """
        递归展开粒子，将中间态替换为它们的衰变产物
        """
        if depth > self.max_depth:
            return particles
      
        expanded = []
      
        for particle in particles:
            if particle.is_stable:
                # 稳定粒子直接添加
                expanded.append(particle)
            else:
                # 生成该粒子的衰变产物
                decay_products = self._generate_decay_products(particle)
              
                # 递归展开衰变产物
                expanded_products = self._expand_particles(decay_products, depth + 1)
              
                # 添加中间态衰变记录
                expanded.append({
                    'particle': particle.name,
                    'decay_products': expanded_products
                })
      
        return expanded
  
    def _extract_final_particles(self, chain: List) -> List[Particle]:
        """
        从完整的衰变链中提取所有最终稳定粒子
        """
        final_particles = []
      
        for item in chain:
            if isinstance(item, Particle):
                final_particles.append(item)
            elif isinstance(item, dict):
                # 递归处理中间态的衰变产物
                final_particles.extend(self._extract_final_particles(item['decay_products']))
      
        return final_particles
  
    def _generate_decay_products(self, parent: Particle) -> List[Particle]:
        """生成衰变产物，支持预设衰变模式和随机生成"""
        # 优先使用预设衰变模式（如果有）
        if parent.branching_fractions and parent.exclusive_branching_fractions:
            return self._select_from_preset_decays(parent)
      
        # 否则随机生成衰变产物
        return self._generate_random_decay(parent)
  
    def _select_from_preset_decays(self, parent: Particle) -> List[Particle]:
        """从预设衰变模式中选择"""
        # 选择衰变模式（根据分支比）
        total_bf = sum(parent.branching_fractions)
        if total_bf <= 0:  # 防止除零错误
            return self._generate_random_decay(parent)
      
        rand_val = random.uniform(0, total_bf)
        cumulative = 0
      
        for i, bf in enumerate(parent.branching_fractions):
            cumulative += bf
            if rand_val <= cumulative:
                # 获取对应的衰变产物
                decay_mode = parent.exclusive_branching_fractions[i]
                return [self.db[name] for name in decay_mode]
      
        # 默认返回第一个衰变模式
        return [self.db[name] for name in parent.exclusive_branching_fractions[0]]
  
    def _generate_random_decay(self, parent: Particle) -> List[Particle]:
        """随机生成符合守恒定律的衰变产物"""
        # 尝试多次寻找有效衰变
        for _ in range(self.max_attempts):
            # 随机决定产物数量 (2-4个粒子)
            num_products = random.randint(2, 4)
            candidates = self._get_possible_products()
          
            # 随机选择产物
            products = random.sample(candidates, num_products)
          
            # 检查守恒
            if self.conservation.check([parent], products):
                return products
      
        # 如果找不到有效衰变，添加默认粒子
        return [Particle("UNKNOWN", {'charge': 0, 'mass': 0, 'is_stable': True})]
  
    def _get_possible_products(self) -> List[Particle]:
        """获取可能的衰变产物候选"""
        # 包括所有粒子和反粒子
        return list(self.db.values())
    
    def process_decay_chain(self, decay_chain: Dict) -> Dict:
        def custom_hash(obj):
            if isinstance(obj, dict):
                # 对字典的键值对进行排序，确保键的顺序一致
                return hash(tuple(sorted((k, custom_hash(v)) for k, v in obj.items())))
            elif isinstance(obj, list):
                # 将列表转换为集合，忽略元素顺序，并确保元素唯一
                return hash(frozenset(custom_hash(v) for v in obj))
            else:
                # 对于其他类型，直接使用内置的 hash() 函数
                return hash(obj)
    
        processed_chain = {}
        for key, value in decay_chain.items():
            processed_chain[key] = custom_hash(value)
        return processed_chain
    
    def format_decay_chain(self, chain: Dict) -> str:
        """格式化衰变链为字符串"""
        # 格式化单个衰变项
        def format_item(item):
            if isinstance(item, Particle):
                return item.name
            elif isinstance(item, dict):
                products = " ".join(format_item(p) for p in item['decay_products'])
                return f"[{item['particle']} -> {products}]"
            return ""
    
        # 初始状态
        initial_str = " ".join(chain['initial_state'])
    
        # 完整衰变链
        chain_str = " ".join(format_item(item) for item in chain['full_chain'])
    
        return f"{initial_str} -> {chain_str}"

def create_particle_database() -> Dict[str, Particle]:
    """创建粒子数据库"""
    db = {}
  
    # 轻子
    db["e⁻"] = Particle("e⁻", {
        'charge': -1, 'is_lepton': True, 'is_stable': True,
        'mass': 0.511, 'quantum_J': 0.5, 'quantum_P': -1
    })
    db["e⁺"] = Particle("e⁺", {
        'charge': 1, 'is_lepton': True, 'is_stable': True,
        'mass': 0.511, 'quantum_J': 0.5, 'quantum_P': -1
    })
    db["μ⁺"] = Particle("μ⁺", {
        'charge': 1, 'is_lepton': True, 'is_stable': True,
        'mass': 105.66, 'lifetime': 2.2e-6
    })
    db["μ⁻"] = Particle("μ⁻", {
        'charge': -1, 'is_lepton': True, 'is_stable': True,
        'mass': 105.66, 'lifetime': 2.2e-6
    })
    # db["ν_e"] = Particle("ν_e", {'charge': 0, 'is_lepton': True, 'mass': 0, 'is_stable': True})
    # db["ν_μ"] = Particle("ν_μ", {'charge': 0, 'is_lepton': True, 'mass': 0, 'is_stable': True})
  
    # 光子
    # db["γ"] = Particle("γ", {
    #     'charge': 0, 'is_boson': True, 'is_stable': True,
    #     'mass': 0, 'quantum_J': 1, 'quantum_P': -1
    # })
  
    # π介子
    db["π⁺"] = Particle("π⁺", {
        'charge': 1, 'is_meson': True, 'is_stable': True,
        'mass': 139.57, 'lifetime': 2.6e-8,
        'quantum_I': 1, 'quantum_J': 0, 'quantum_P': -1,
        #'branching_fractions': [0.999877, 0.000123],
        #'exclusive_branching_fractions': [["μ⁺", "ν_μ"], ["e⁺", "ν_e"]]
    })
    db["π⁻"] = Particle("π⁻", {
        'charge': -1, 'is_meson': True, 'is_stable': True,
        'mass': 139.57, 'lifetime': 2.6e-8,
        'quantum_I': 1, 'quantum_J': 0, 'quantum_P': -1
    })
    db["π⁰"] = Particle("π⁰", {
        'charge': 0, 'is_meson': True, 'is_stable': True,
        'mass': 134.98, 'lifetime': 8.4e-17,
        'quantum_I': 1, 'quantum_J': 0, 'quantum_P': -1,
        # 'branching_fractions': [0.988, 0.012],
        # 'exclusive_branching_fractions': [["γ", "γ"], ["e⁺", "e⁻", "γ"]]
    })
  
    # K介子
    db["K⁺"] = Particle("K⁺", {
        'charge': 1, 'strangeness': 1, 'is_meson': True, 'is_stable': True,
        'mass': 493.677, 'lifetime': 1.24e-8,
        'quantum_J': 0, 'quantum_P': -1,
        # 'branching_fractions': [0.6355, 0.2114, 0.0557],
        # 'exclusive_branching_fractions': [
        #     ["μ⁺", "ν_μ"], ["π⁺", "π⁰"], ["π⁺", "π⁺", "π⁻"]
        # ]
    })
    db["K⁻"] = Particle("K⁻", {
        'charge': -1, 'strangeness': -1, 'is_meson': True, 'is_stable': True,
        'mass': 493.677, 'lifetime': 1.24e-8
    })
  
    # φ介子 (中间态)
    db["φ"] = Particle("φ", {
        'charge': 0, 'strangeness': 0, 'is_meson': True, 'is_stable': False,
        'mass': 1019.461, 'width': 4.266, 'is_intermediate': True,
        'quantum_J': 1, 'quantum_P': -1,
        'branching_fractions': [0.499],
        'exclusive_branching_fractions': [
            ["K⁺", "K⁻"]
        ]
    })
  
    # J/ψ粒子 (中间态)
    db["J/ψ"] = Particle("J/ψ", {
        'charge': 0, 'is_meson': True, 'is_stable': False,
        'mass': 3096.9, 'width': 0.0929, 'is_intermediate': True,
        'quantum_J': 1, 'quantum_P': -1,
        'branching_fractions': [0.0593, 0.057],
        'exclusive_branching_fractions': [
            ["e⁺", "e⁻"], ["μ⁺", "μ⁻"]
        ]
    })
  
    # 质子
    db["p"] = Particle("p", {
        'charge': 1, 'baryon_number': 1, 'is_baryon': True, 'is_stable': True,
        'mass': 938.272, 'quantum_J': 0.5, 'quantum_P': 1
    })
  
    # 中子
    db["n"] = Particle("n", {
        'charge': 0, 'baryon_number': 1, 'is_baryon': True, 'is_stable': True,
        'mass': 939.565, 'lifetime': 880.0,
        'quantum_J': 0.5, 'quantum_P': 1
    })
  
    # Λ重子
    db["Λ⁰"] = Particle("Λ⁰", {
        'charge': 0, 'strangeness': -1, 'baryon_number': 1, 'is_baryon': True, 'is_stable': True,
        'mass': 1115.683, 'lifetime': 2.632e-10,
        'quantum_J': 0.5, 'quantum_P': 1,
        # 'branching_fractions': [0.639, 0.358],
        # 'exclusive_branching_fractions': [["p", "π⁻"], ["n", "π⁰"]]
    })
  
    # 其他粒子
  
    return db



if __name__ == "__main__":
    # 创建粒子数据库
    particle_db = create_particle_database()
  
    # 设置守恒规则
    conservation_rules = {
        'charge': True,
        'energy': True,
        'strangeness': True,
        'baryon_number': True,
        'lepton_number': True
    }
  
    # 创建衰变链生成器
    ecms = 3097.0 # 单位为MeV
    generator = DecayChainGenerator(particle_db, conservation_rules, ecms, max_depth=10, max_final_particles=8)
  
    # 示例1: e⁺ e⁻ 对撞湮灭
    for _ in range(10):
        chain = generator.generate(["e⁺", "e⁻"])
        print(chain)
  
    # # 示例2: φ介子衰变（中间态）
    # chain = generator.generate(["φ"])
    # print(chain)
  
    # # 示例3: 包含中间态的复杂衰变
    # chain = generator.generate(["J/ψ"])
    # print(chain)
  
    # # 示例4: 多粒子初始状态
    # chain = generator.generate(["π⁺", "p"])
    # print(chain)
  
    # # 示例5: 包含奇异数的衰变
    # chain = generator.generate(["Λ⁰"])
    # print(chain)