using ChainRulesCore: ChainRulesCore
using ChainRules: ChainRules, rrule
using IRTools: varargs!, inlineable!, pis!, slots!
using IRTools.Inner: argnames!, update!

include("tools.jl")
include("emit.jl")
include("reverse.jl")
include("chainrules.jl")
include("interface.jl")