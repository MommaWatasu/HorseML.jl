using Tracker: TrackedArray, TrackedReal

"""
    isexpr(x, ts...)
Convenient way to test the type of a Julia expression.
Expression heads and types are supported, so for example
you can call
    isexpr(expr, String, :string)
to pick up on all string-like expressions.
"""
isexpr(x::Expr) = true
isexpr(x) = false
isexpr(x::Expr, ts...) = x.head in ts
isexpr(x, ts...) = any(T->isa(T, Type) && isa(x, T), ts)

isline(ex) = isexpr(ex, :line) || isa(ex, LineNumberNode)

iscall(ex, f) = isexpr(ex, :call) && ex.args[1] == f

macro __new__(T, args...)
  esc(Expr(:new, T, args...))
end

macro __splatnew__(T, args)
  esc(Expr(:splatnew, T, args))
end

@inline __new__(T, args...) = @__splatnew__(T, args)
@inline __splatnew__(T, args) = @__splatnew__(T, args)

unwrap(x::Union{TrackedArray,TrackedReal}) = Tracker.data(x)