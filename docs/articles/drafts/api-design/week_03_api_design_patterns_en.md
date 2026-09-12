# Why I Intentionally Repeated Code in foampilot: Intentional Repetition as an API Virtue

*In software engineering, we're taught that DRY (Don't Repeat Yourself) is sacred. Yet in foampilot, I deliberately repeated `write()` across 8 classes. Here's why it's a design choice — and why it makes the API more predictable.*

---

## Introduction: The DRY Heresy

In software development, few principles are as sacred as **DRY**: *Don't Repeat Yourself*. "If you're repeating yourself, factorize." It's the first rule we learn, and it's generally correct.

But in foampilot, I made the opposite choice. I deliberately repeated design patterns — notably the `write()` method — across multiple classes. And it is, I believe, one of the best design decisions in the project.

Why? Because in scientific API design, **predictability is worth more than brevity**.

---

## The Trap of Excessive Abstraction

Let's take a concrete example. At the beginning of the project, I could have factorized file writing like this:

```python
# ❌ What foampilot could have done
class CaseWriter:
    def write_solver_files(self, solver):
        # Write solver files
        pass
    
    def write_mesh_files(self, mesher):
        # Write mesh files
        pass
    
    def write_boundary_files(self, boundary):
        # Write boundary files
        pass
```

**Problem**: the user must learn 3 different method names to do the same fundamental thing — write files. The documentation grows longer, the API surface expands, and every new user must wonder: *"Is it `write_solver_files()` or `write_case()`?"*

This is what I call **invisible abstraction**: it looks elegant in the code, but it makes the API harder to learn and use.

---

## The Universal `write()`: A Predictable Interface

In foampilot, I chose simplicity:

```python
# ✅ What foampilot actually does
class OpenFOAMFile:
    def write_file(self, filepath):
        # Write dictionary

class SystemDirectory:
    def write(self):
        self.controlDict.write(system_path / 'controlDict')
        self.fvSchemes.write(system_path / 'fvSchemes')

class ConstantDirectory:
    def write(self):
        self.transportProperties.write(constant_path / 'transportProperties')
        self.turbulenceProperties.write(constant_path / 'turbulenceProperties')

class Meshing:
    def write(self):
        self.mesher.write()
```

Eight classes. Eight `write()` methods. Same signature. Same semantics: *"persist yourself to disk"*.

**The advantage?** You never have to wonder *"how do I write this object?"*. The answer is always the same:

```python
for component in [solver.system, solver.constant, meshing, boundary]:
    component.write()
```

This is the **Command Pattern** in its purest form: each object encapsulates the action of persisting itself. The repetition of `write()` is not a flaw — it's an **interface contract** that makes the API learnable in 10 minutes.

---

## `apply_condition_with_wildcard`: One Signature to Rule Them All

Same logic for boundary conditions. Instead of exposing 15 different methods (`set_velocity_inlet()`, `set_pressure_outlet()`, `set_wall()`, `set_symmetry()`, etc.), I created a single method:

```python
# foampilot/boundaries/boundaries_dict.py
def apply_condition_with_wildcard(self, pattern: str, condition_type: str, **kwargs):
    for boundary in self.fields[next(iter(self.fields))].keys():
        if re.match(pattern, boundary):
            self.set_condition(boundary, condition_type, **kwargs)
```

One signature for all conditions. The user learns one method, and can apply any condition:

```python
solver.boundary.apply_condition_with_wildcard("inlet", "velocityInlet", velocity=(10, 0, 0))
solver.boundary.apply_condition_with_wildcard("outlet", "pressureOutlet")
solver.boundary.apply_condition_with_wildcard("walls", "wall", friction=True)
```

**API surface reduction**: from 15 methods to 1. **Cognitive load reduction**: the user no longer has to remember the exact name of each method.

---

## Why Repetition is a Virtue Here

In most projects, code duplication is an anti-pattern. In foampilot, it serves three purposes:

### 1. Discoverability

When every object has a `write()`, you don't have to consult documentation to find the right method name. The interface is **self-documenting**.

### 2. Composability

Uniform interfaces enable generic algorithms. You can write:

```python
def write_all_case_files(objects):
    for obj in objects:
        obj.write()
```

This function works on any combination of foampilot objects, because they all share the same protocol. This is the power of **polymorphism by interface** — no type checking (`isinstance`), no adapting each object.

### 3. Reduced Cognitive Load

A CFD engineer already has a lot to remember: numerical schemes, turbulence models, meshing, boundary conditions. An API where everything works the same way eliminates one more mental variable.

> **"The best API is the one you don't have to learn — because everything works as expected."**

---

## When to Factorize, When to Repeat

This design choice isn't universal. Here's a decision table:

| Situation | Factorize | Repeat |
|-----------|-----------|--------|
| Identical business logic | ✅ | ❌ |
| Public user interface | ❌ | ✅ |
| Complex algorithm | ✅ | ❌ |
| Public entry point | ❌ | ✅ |
| Data validation | ✅ | ❌ |
| Persistence (write/save) | ❌ | ✅ |

**Simple rule**: if it concerns the **user** (what they see, what they call), prefer repetition. If it concerns the **implementation** (what the code does), factorize.

---

## Impact on Developer Experience

I've measured the impact of this choice in several ways:

- **API learning time**: ~10 minutes for a new user
- **Number of `write()` methods**: 8 classes, 8 methods, same name
- **Number of `apply_condition*` methods**: 1 method, 15 supported conditions
- **"Which method should I use?" error rate**: near zero

User feedback confirms: *"I saw the documentation, I saw the code, and I understood immediately how it works."*

This is **DX (Developer Experience)**: making the tool so predictable that it disappears. The user no longer thinks about the API — they think about their simulation.

---

## Conclusion: Predictability as a Virtue

In scientific tool design, simplicity isn't a luxury — it's a necessity. A CFD engineer using your API does it to solve a physical problem, not to learn a new interface.

By intentionally repeating `write()` and `apply_condition_with_wildcard`, I created an API where:
- **Everything is written with `.write()`**
- **Every condition is applied with `.apply_condition_with_wildcard()`**

No surprises. No need to consult documentation for each class. Just two patterns to remember, and that's it.

In the next article, I'll show how this predictability enables visualizing OpenFOAM directly from Python — no conversion, no `foamToVTK`, no pain.

**Resources:**
- Repository: [github.com/stevendaix/foampilot](https://github.com/stevendaix/foampilot)
- Documentation: [stevendaix.github.io/foampilot](https://stevendaix.github.io/foampilot/)

---

*Article under improvement — version 1.0*
