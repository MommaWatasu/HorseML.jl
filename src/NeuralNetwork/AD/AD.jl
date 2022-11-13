using ChainRulesCore: ChainRulesCore
using ChainRules: rrule
using IRTools: varargs!, inlineable!, pis!, slots!
using IRTools.Inner: argnames!, update!

include("reverse.jl")
include("chainrules.jl")
include("interface.jl")