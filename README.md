# AutoDiff: Higher-Order Automatic Differentiation in Idris

This library implements automatic differentiation in Idris with support for both forward and backward (reverse) mode differentiation. It can compute derivatives of any order for a wide range of mathematical functions.

## Overview

Automatic differentiation (AD) is a set of techniques to numerically evaluate the derivative of a function specified by a computer program. It's different from both symbolic differentiation and numerical differentiation (finite differences) by providing efficient, accurate derivatives without the overhead of symbolic manipulation or the numerical errors of finite differences.

This implementation features:
- **Forward mode** differentiation for first derivatives
- **Backward mode** differentiation for arbitrary-order derivatives
- Entirely implemented in pure Idris with no external dependencies
- Custom data structures to ensure compatibility across Idris versions

## Key Features

1. **Dual Approach**: Implements both forward-mode AD (efficient for functions with few inputs) and backward-mode AD (efficient for functions with many outputs)

2. **Higher-Order Derivatives**: Support for computing derivatives of any order in backward mode

3. **Extensive Function Support**: Works with:
   - Polynomial functions
   - Trigonometric functions (sin, cos)
   - Exponential and logarithmic functions
   - Compositions and combinations of the above

4. **Type Safety**: Leverages Idris's strong type system to prevent runtime errors

## Quick Start

### Installation

Clone this repository and compile with Idris 2:

```bash
idris2 -o autodiff AutoDiff.idr
```

### Basic Usage

To run the included examples:

```bash
./build/exec/autodiff
```

### Using the Library

#### Forward Mode (First Derivatives)

```idris
import AutoDiff

-- Define a function using Forward type
myFunc : Forward -> Forward
myFunc x@(MkForward val ds) = 
  case ds of
    Nil => MkForward (val * val + 3.0 * val) Nil  -- Function value
    (d :: _) => 
      let deriv = (2.0 * val + 3.0) * d  -- First derivative
      in MkForward (val * val + 3.0 * val) (deriv :: Nil)

-- Calculate derivative at x=2.0
derivative = forward_derivative myFunc 2.0 1  -- Get first derivative
```

#### Backward Mode (Arbitrary-Order Derivatives)

```idris
import AutoDiff

-- Define a function using Expr type
myFunc : Expr -> Expr
myFunc x = AddExpr (MulExpr x x) (MulExpr (ConstExpr 3.0) x)

-- Calculate derivatives at x=2.0
firstDeriv = backward_derivative myFunc 2.0 1   -- First derivative
secondDeriv = backward_derivative myFunc 2.0 2  -- Second derivative
thirdDeriv = backward_derivative myFunc 2.0 3   -- Third derivative
```

## Example Functions

The library includes 10 example functions demonstrating various use cases:

1. **Polynomial**: f(x) = x² + 3x + 2
2. **Trigonometric Product**: f(x) = sin(x) × cos(x)
3. **Compound Function**: f(x) = e^(x²) / (1 + x²)
4. **Higher-degree Polynomial**: f(x) = x³ - 5x² + 7x - 3
5. **Composition of Functions**: f(x) = sin(x²)
6. **Logarithmic Rational Function**: f(x) = log(x) / x
7. **Nested Trigonometric Functions**: f(x) = cos(sin(x))
8. **Exponential of Trigonometric Sum**: f(x) = exp(sin(x) + cos(x))
9. **Rational Function**: f(x) = x / (1 + x²)²
10. **Gaussian-related Function**: f(x) = x * exp(-x²/2)

## Technical Details

### Data Structures

- `DList`: Custom list implementation for derivative values
- `Forward`: Represents a value and its derivatives in forward mode
- `Expr`: Expression tree for symbolic differentiation in backward mode

### Core Functions

- `forward_derivative`: Compute nth derivative using forward mode
- `backward_derivative`: Compute nth derivative using backward mode
- `gradToExpr`: Convert derivatives to expressions for higher-order differentiation

### Implementation Notes

- Forward mode currently optimized for first derivatives
- Backward mode supports derivatives of any order via recursive symbolic differentiation
- Custom implementations for arithmetic, trigonometric, and transcendental functions

## Mathematical Background

### Forward Mode

Based on dual numbers, which extend real numbers with an infinitesimal ε where ε² = 0. Operations are defined to track both values and derivatives simultaneously.

### Backward Mode

Based on building a computational graph and applying the chain rule recursively. This allows efficient calculation of higher-order derivatives by differentiating the derivative expressions.

## Future Enhancements

- Support for higher-order derivatives in forward mode
- Vector and matrix operations for multivariate differentiation
- Integration with numerical optimization algorithms
- Performance optimizations for large-scale computations
