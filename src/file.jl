# when iterating through rows that are views into `Column2`s,
# it's efficient to keep track of which column chunk we're iterating
# through, hence ThreadedIterationState
mutable struct ThreadedIterationState
    row::Int64
    array_index::Int64
    array_i::Int64
    array_len::Int64
    array_lens::Vector{Int64}
end

# normal single-threaded Column with a single `tape` that holds
# our results from parsing
struct Column{T, P} <: AbstractVector{T}
    tape::Vector{UInt64}
    len::Int
    e::UInt8
    catg::Bool
    refs::Union{Vector{String}, Nothing}
    buf::Vector{UInt8}
    sentinel::UInt64
end

# Column2 holds a `Column` per thread that was parsed
# each `Column` has a tape for a chunk of the file
struct Column2{T, P} <: AbstractVector{T}
    columns::Vector{Column{T, P}}
    len::Int
end

_eltype(::Type{T}) where {T} = T
_eltype(::Type{PooledString}) = String
_eltype(::Type{Union{PooledString, Missing}}) = Union{String, Missing}

Base.size(c::Union{Column, Column2}) = (c.len,)
Base.IndexStyle(::Type{<:Column}) = Base.IndexLinear()
Base.IndexStyle(::Type{<:Column2}) = Base.IndexLinear()

# getindex definitions in tables.jl

# a Row "view" type for iterating `CSV.File`
struct Row{threaded} <: Tables.AbstractRow
    names::Vector{Symbol}
    columns::Vector{AbstractVector}
    lookup::Dict{Symbol, AbstractVector}
    row::Int64
    array_index::Int64
    array_i::Int64
end

getnames(r::Row) = getfield(r, :names)
getcolumn(r::Row, col::Int) = getfield(r, :columns)[col]
getcolumn(r::Row, col::Symbol) = getfield(r, :lookup)[col]
getrow(r::Row) = getfield(r, :row)
getarrayindex(r::Row) = getfield(r, :array_index)
getarrayi(r::Row) = getfield(r, :array_i)

Tables.columnnames(r::Row) = getnames(r)

# main structure when parsing an entire file and inferring column types
struct File{threaded} <: AbstractVector{Row{threaded}}
    name::String
    names::Vector{Symbol}
    types::Vector{Type}
    rows::Int64
    cols::Int64
    columns::Vector{AbstractVector}
    lookup::Dict{Symbol, AbstractVector}
end

getname(f::File) = getfield(f, :name)
getnames(f::File) = getfield(f, :names)
gettypes(f::File) = getfield(f, :types)
getrows(f::File) = getfield(f, :rows)
getcols(f::File) = getfield(f, :cols)
getcolumns(f::File) = getfield(f, :columns)
getlookup(f::File) = getfield(f, :lookup)
getcolumn(f::File, col::Int) = getfield(f, :columns)[col]
getcolumn(f::File, col::Symbol) = getfield(f, :lookup)[col]

function Base.show(io::IO, f::File)
    println(io, "CSV.File(\"$(getname(f))\"):")
    println(io, "Size: $(getrows(f)) x $(getcols(f))")
    show(io, Tables.schema(f))
end

Base.IndexStyle(::Type{File}) = Base.IndexLinear()
Base.eltype(f::File{threaded}) where {threaded} = Row{threaded}
Base.size(f::File) = (getrows(f),)

function allocate(rowsguess, ncols, types)
    columns = AbstractVector[allocate(types[i], rowsguess) for i = 1:ncols]
    poslens = Vector{Vector{PosLen}}(undef, ncols)
    for i = 1:ncols
        if !user(types[i])
            poslens[i] = Vector{PosLen}(undef, rowsguess)
        end
    end
    return columns, poslens
end

"""
    CSV.File(source; kwargs...) => CSV.File

Read a UTF-8 CSV input (a filename given as a String or FilePaths.jl type, or any other IO source), returning a `CSV.File` object.

Opens the file and uses passed arguments to detect the number of columns and column types, unless column types are provided
manually via the `types` keyword argument. Note that passing column types manually can increase performance and reduce the
memory use for each column type provided (column types can be given as a `Vector` for all columns, or specified per column via
name or index in a `Dict`). For text encodings other than UTF-8, see the [StringEncodings.jl](https://github.com/JuliaStrings/StringEncodings.jl)
package for re-encoding a file or IO stream.
The returned `CSV.File` object supports the [Tables.jl](https://github.com/JuliaData/Tables.jl) interface
and can iterate `CSV.Row`s. `CSV.Row` supports `propertynames` and `getproperty` to access individual row values. `CSV.File`
also supports entire column access like a `DataFrame` via direct property access on the file object, like `f = CSV.File(file); f.col1`.
Note that duplicate column names will be detected and adjusted to ensure uniqueness (duplicate column name `a` will become `a_1`).
For example, one could iterate over a csv file with column names `a`, `b`, and `c` by doing:

```julia
for row in CSV.File(file)
    println("a=\$(row.a), b=\$(row.b), c=\$(row.c)")
end
```

By supporting the Tables.jl interface, a `CSV.File` can also be a table input to any other table sink function. Like:

```julia
# materialize a csv file as a DataFrame, without copying columns from CSV.File; these columns are read-only
df = CSV.File(file) |> DataFrame!

# load a csv file directly into an sqlite database table
db = SQLite.DB()
tbl = CSV.File(file) |> SQLite.load!(db, "sqlite_table")
```

Supported keyword arguments include:
* File layout options:
  * `header=1`: the `header` argument can be an `Int`, indicating the row to parse for column names; or a `Range`, indicating a span of rows to be concatenated together as column names; or an entire `Vector{Symbol}` or `Vector{String}` to use as column names; if a file doesn't have column names, either provide them as a `Vector`, or set `header=0` or `header=false` and column names will be auto-generated (`Column1`, `Column2`, etc.)
  * `normalizenames=false`: whether column names should be "normalized" into valid Julia identifier symbols; useful when iterating rows and accessing column values of a row via `getproperty` (e.g. `row.col1`)
  * `datarow`: an `Int` argument to specify the row where the data starts in the csv file; by default, the next row after the `header` row is used. If `header=0`, then the 1st row is assumed to be the start of data
  * `skipto::Int`: similar to `datarow`, specifies the number of rows to skip before starting to read data
  * `footerskip::Int`: number of rows at the end of a file to skip parsing
  * `limit`: an `Int` to indicate a limited number of rows to parse in a csv file; use in combination with `skipto` to read a specific, contiguous chunk within a file
  * `transpose::Bool`: read a csv file "transposed", i.e. each column is parsed as a row
  * `comment`: rows that begin with this `String` will be skipped while parsing
  * `use_mmap::Bool=!Sys.iswindows()`: whether the file should be mmapped for reading, which in some cases can be faster
  * `ignoreemptylines::Bool=false`: whether empty rows/lines in a file should be ignored (if `false`, each column will be assigned `missing` for that empty row)
  * `threaded::Bool`: whether parsing should utilize multiple threads; by default threads are used on large enough files, but isn't allowed when `transpose=true` or when `limit` is used; only available in Julia 1.3+
  * `select`: an `AbstractVector` of `Int`, `Symbol`, `String`, or `Bool`, or a "selector" function of the form `(i, name) -> keep::Bool`; only columns in the collection or for which the selector function returns `true` will be parsed and accessible in the resulting `CSV.File`. Invalid values in `select` are ignored.
  * `drop`: inverse of `select`; an `AbstractVector` of `Int`, `Symbol`, `String`, or `Bool`, or a "drop" function of the form `(i, name) -> drop::Bool`; columns in the collection or for which the drop function returns `true` will ignored in the resulting `CSV.File`. Invalid values in `drop` are ignored.
* Parsing options:
  * `missingstrings`, `missingstring`: either a `String`, or `Vector{String}` to use as sentinel values that will be parsed as `missing`; by default, only an empty field (two consecutive delimiters) is considered `missing`
  * `delim=','`: a `Char` or `String` that indicates how columns are delimited in a file; if no argument is provided, parsing will try to detect the most consistent delimiter on the first 10 rows of the file
  * `ignorerepeated::Bool=false`: whether repeated (consecutive) delimiters should be ignored while parsing; useful for fixed-width files with delimiter padding between cells
  * `quotechar='"'`, `openquotechar`, `closequotechar`: a `Char` (or different start and end characters) that indicate a quoted field which may contain textual delimiters or newline characters
  * `escapechar='"'`: the `Char` used to escape quote characters in a quoted field
  * `dateformat::Union{String, Dates.DateFormat, Nothing}`: a date format string to indicate how Date/DateTime columns are formatted for the entire file
  * `decimal='.'`: a `Char` indicating how decimals are separated in floats, i.e. `3.14` used '.', or `3,14` uses a comma ','
  * `truestrings`, `falsestrings`: `Vectors of Strings` that indicate how `true` or `false` values are represented; by default only `true` and `false` are treated as `Bool`
* Column Type Options:
  * `type`: a single type to use for parsing an entire file; i.e. all columns will be treated as the same type; useful for matrix-like data files
  * `types`: a Vector or Dict of types to be used for column types; a Dict can map column index `Int`, or name `Symbol` or `String` to type for a column, i.e. Dict(1=>Float64) will set the first column as a Float64, Dict(:column1=>Float64) will set the column named column1 to Float64 and, Dict("column1"=>Float64) will set the column1 to Float64; if a `Vector` if provided, it must match the # of columns provided or detected in `header`
  * `typemap::Dict{Type, Type}`: a mapping of a type that should be replaced in every instance with another type, i.e. `Dict(Float64=>String)` would change every detected `Float64` column to be parsed as `String`
  * `pool::Union{Bool, Float64}=0.1`: if `true`, *all* columns detected as `String` will be internally pooled; alternatively, the proportion of unique values below which `String` columns should be pooled (by default 0.1, meaning that if the # of unique strings in a column is under 10%, it will be pooled)
  * `categorical::Bool=false`: whether pooled columns should be copied as CategoricalArray instead of PooledArray; note that in `CSV.read`, by default, columns are not copied, so pooled columns will have type `CSV.Column{String, PooledString}`; to get `CategoricalArray` columns, also pass `copycols=true`
  * `strict::Bool=false`: whether invalid values should throw a parsing error or be replaced with `missing`
  * `silencewarnings::Bool=false`: if `strict=false`, whether invalid value warnings should be silenced
"""
function File(source;
    # file options
    # header can be a row number, range of rows, or actual string vector
    header::Union{Integer, Vector{Symbol}, Vector{String}, AbstractVector{<:Integer}}=1,
    normalizenames::Bool=false,
    # by default, data starts immediately after header or start of file
    datarow::Integer=-1,
    skipto::Union{Nothing, Integer}=nothing,
    footerskip::Integer=0,
    limit::Integer=typemax(Int64),
    transpose::Bool=false,
    comment::Union{String, Nothing}=nothing,
    use_mmap::Bool=!Sys.iswindows(),
    ignoreemptylines::Bool=false,
    threaded::Union{Bool, Nothing}=nothing,
    select=nothing,
    drop=nothing,
    # parsing options
    missingstrings=String[],
    missingstring="",
    delim::Union{Nothing, Char, String}=nothing,
    ignorerepeated::Bool=false,
    quotechar::Union{UInt8, Char}='"',
    openquotechar::Union{UInt8, Char, Nothing}=nothing,
    closequotechar::Union{UInt8, Char, Nothing}=nothing,
    escapechar::Union{UInt8, Char}='"',
    dateformat::Union{String, Dates.DateFormat, Nothing}=nothing,
    dateformats::Union{AbstractDict, Nothing}=nothing,
    decimal::Union{UInt8, Char}=UInt8('.'),
    truestrings::Union{Vector{String}, Nothing}=["true", "True", "TRUE"],
    falsestrings::Union{Vector{String}, Nothing}=["false", "False", "FALSE"],
    # type options
    type=nothing,
    types=nothing,
    typemap::Dict=Dict{Type, Type}(),
    categorical::Union{Bool, Real}=false,
    pool::Union{Bool, Real}=0.1,
    strict::Bool=false,
    silencewarnings::Bool=false,
    debug::Bool=false,
    parsingdebug::Bool=false,)

    h = Header(source, header, normalizenames, datarow, skipto, footerskip, limit, transpose, comment, use_mmap, ignoreemptylines, threaded, select, drop, missingstrings, missingstring, delim, ignorerepeated, quotechar, openquotechar, closequotechar, escapechar, dateformat, dateformats, decimal, truestrings, falsestrings, type, types, typemap, categorical, pool, strict, silencewarnings, debug, parsingdebug, false)
    rowsguess, ncols, buf, len, datapos, options, coloptions, positions, types, pool, categorical = h.rowsguess, h.cols, h.buf, h.len, h.datapos, h.options, h.coloptions, h.positions, h.types, h.pool, h.categorical
    # determine if we can use threads while parsing
    if threaded === nothing && VERSION >= v"1.3-DEV" && Threads.nthreads() > 1 && !transpose && limit == typemax(Int64) && rowsguess > Threads.nthreads() && (rowsguess * ncols) >= 5_000
        threaded = true
    elseif threaded === true
        if VERSION < v"1.3-DEV"
            @warn "incompatible julia version for `threaded=true`: $VERSION, requires >= v\"1.3\", setting `threaded=false`"
            threaded = false
        elseif transpose
            @warn "`threaded=true` not supported on transposed files"
            threaded = false
        elseif limit != typemax(Int64)
            @warn "`threaded=true` not supported when limiting # of rows"
            threaded = false
        end
    end

    # we now do our parsing pass over the file, starting at datapos
    # `tapes` are Vector{UInt64} for each column, with length == # of estimated rows
    # all types are treated as UInt64 for type stability while looping over columns and for making "recoding" more convenient
    # (e.g. when we "promote" a column type from Int => Float64)
    # we have a sentinel value for each value to signal a `missing` value; this sentinel value is tracked internally and adjusted
    # automatically if an actual value of the sentinel value is parsed
    # for strings, we have the following encoding of a single UInt64:
        # leftmost bit indicates a sentinel value (`missing`) was detected while parsing
        # 2nd leftmost bit indicates if a field was quoted and included escape chararacters (will have to be unescaped later)
        # 42 bits for position (allows for maximum file size of ~4TB)
        # 20 bits for field length (allows for maximum field size of ~1M)
    # the `poslens` are also Vector{UInt64} allocated for each column where the type must be detected; it stores the "string value" UInt64
    # of the cell, which allows promoting any column to a string later if needed
    # if a column type if promoted to string, the values are stored in the corresponding `tape` instead of `poslen`
    if threaded === true
        # multithread
        rows, tapes, refs, typecodes, intsentinels = multithreadparse(typecodes, buf, datapos, len, options, coloptions, rowsguess, pool, ncols, typemap, limit, debug)
        finalrows = sum(rows)
    else
        # intsentinels = fill(INT_SENTINEL, ncols)
        tapes, poslens = allocate(rowsguess, ncols, types)
        refs = Vector{Dict{String, UInt32}}(undef, ncols)
        lastrefs = zeros(UInt32, ncols)
        t = Base.time()
        rows, tapes, poslens = parsetape(Val(transpose), ncols, typemap, tapes, poslens, buf, datapos, len, limit, positions, pool, refs, lastrefs, rowsguess, types, debug, options, coloptions)
        finalrows = rows
        debug && println("time for initial parsing to tape: $(Base.time() - t)")
    end
    finaltypes = types
    debug && println("types after parsing: $finaltypes, pool = $pool")
    finalrefs = Vector{Union{Vector{String}, Nothing}}(undef, ncols)
    if pool > 0.0
        for i = 1:ncols
            if isassigned(refs, i)
                finalrefs[i] = map(x->x[1], sort!(collect(refs[i]), by=x->x[2]))
            elseif pooled(typebits(types[i]))
                # case where user manually specified types, but no rows were parsed
                # so the refs never got initialized; initialize them here to empty
                finalrefs[i] = Vector{String}[]
            else
                finalrefs[i] = nothing
            end
            resize!(tapes[i], rows)
        end
    else
        fill!(finalrefs, nothing)
    end
    if threaded === true
        columns = Union{Column, Column2}[Column2{_eltype(finaltypes[i]), finaltypes[i]}([Column{_eltype(finaltypes[i]), finaltypes[i]}(tapes[j][i], rows[j], h.e, categorical, finalrefs[i], buf, finaltypes[i] >: Int64 ? uint64(intsentinels[i]) : sentinelvalue(Base.nonmissingtype(finaltypes[i]))) for j = 1:Threads.nthreads()], finalrows) for i = 1:ncols]
    else
        # columns = Union{Column, Column2}[Column{_eltype(finaltypes[i]), finaltypes[i]}(tapes[i], rows, h.e, categorical, finalrefs[i], buf, finaltypes[i] >: Int64 ? uint64(intsentinels[i]) : sentinelvalue(Base.nonmissingtype(finaltypes[i]))) for i = 1:ncols]
        columns = tapes
    end
    deleteat!(h.names, h.todrop)
    deleteat!(finaltypes, h.todrop)
    ncols -= length(h.todrop)
    deleteat!(columns, h.todrop)
    lookup = Dict(k => v for (k, v) in zip(h.names, columns))
    return File{something(threaded, false)}(h.name, h.names, finaltypes, finalrows, ncols, columns, lookup)
end

function multithreadparse(typecodes, buf, datapos, len, options, coloptions, rowsguess, pool, ncols, typemap, limit, debug)
    N = Threads.nthreads()
    chunksize = div(len - datapos, N)
    ranges = [datapos, (datapos + chunksize * i for i = 1:N)...]
    ranges[end] = len
    debug && println("initial byte positions before adjusting for start of rows: $ranges")
    findrowstarts!(buf, len, options, ranges, ncols)
    rowchunkguess = cld(rowsguess, N)
    debug && println("parsing using $N threads: $rowchunkguess rows chunked at positions: $ranges")
    perthreadrows = Vector{Int}(undef, N)
    perthreadtapes = Vector{Vector{Vector{UInt64}}}(undef, N)
    perthreadposlens = Vector{Vector{Vector{UInt64}}}(undef, N)
    perthreadrefs = Vector{Vector{Dict{String, UInt64}}}(undef, N)
    perthreadlastrefs = Vector{Vector{UInt64}}(undef, N)
    perthreadtypecodes = [copy(typecodes) for i = 1:N]
    perthreadintsentinels = Vector{Vector{Int64}}(undef, N)
    @sync for i = 1:N
@static if VERSION >= v"1.3-DEV"
        Threads.@spawn begin
            tt = Base.time()
            tl_refs = Vector{Dict{String, UInt64}}(undef, ncols)
            tl_lastrefs = zeros(UInt64, ncols)
            tl_tapes, tl_poslens = allocate(rowchunkguess, ncols, typecodes)
            tl_intsentinels = fill(INT_SENTINEL, ncols)
            tl_datapos = ranges[i]
            tl_len = ranges[i + 1] - (i != N)
            tl_rows, tl_tapes, tl_poslens = parsetape(Val(false), ncols, typemap, tl_tapes, tl_poslens, buf, tl_datapos, tl_len, limit, Int64[], pool, tl_refs, tl_lastrefs, rowchunkguess, perthreadtypecodes[i], tl_intsentinels, debug, options, coloptions)
            debug && println("thread = $(Threads.threadid()): time for parsing: $(Base.time() - tt)")
            perthreadrows[i] = tl_rows
            perthreadtapes[i] = tl_tapes
            perthreadposlens[i] = tl_poslens
            perthreadrefs[i] = tl_refs
            perthreadlastrefs[i] = tl_lastrefs
            perthreadintsentinels[i] = tl_intsentinels
        end
end # @static if VERSION >= v"1.3-DEV"
    end
    intsentinels = perthreadintsentinels[1]
    anyintrecode = false
    # promote typecodes from each thread
    for col = 1:ncols
        for i = 1:N
            @inbounds typecodes[col] = promote_typecode(typecodes[col], perthreadtypecodes[i][col])
            @inbounds if perthreadintsentinels[N][col] != INT_SENTINEL
                intsentinels[col] = perthreadintsentinels[N][col]
                anyintrecode = true
            end
        end
    end
    # if we need to recode any int column sentinels, we need to check that all the other threads
    # don't already have the chosen int sentinel
    if anyintrecode
        for col = 1:ncols
            while true
                foundsent = false
                intsent = uint64(intsentinels[col])
                for i = 1:N
                    if uint64(perthreadintsentinels[i][col]) != intsent
                        tape = perthreadtapes[i][col]
                        for j = 1:perthreadrows[i]
                            @inbounds z = tape[j]
                            if z == intsent
                                foundsent = true
                                break
                            end
                        end
                        foundsent && break
                    end
                end
                if foundsent
                    intsentinels[col] = sentinelvalue(Int64)
                else
                    break
                end
            end
        end
    end
    # merge refs for pooled columns from each thread and recode if needed
    # take care of any column promoting that needs to happen as well between threads
    refs = Vector{Dict{String, UInt64}}(undef, ncols)
    lastrefs = zeros(UInt64, ncols)
    for i = 1:N
        tlrefs = perthreadrefs[i]
        tllastrefs = perthreadlastrefs[i]
        tltypecodes = perthreadtypecodes[i]
        tltapes = perthreadtapes[i]
        tlposlens = perthreadposlens[i]
        tlrows = perthreadrows[i]
        for col = 1:ncols
            @inbounds T = typecodes[col]
            @inbounds TL = tltypecodes[col]
            if T == MISSINGTYPE
                unset!(tltapes, col, tlrows, 1)
                tltapes[col] = UInt64[]
            elseif !stringtype(TL) && stringtype(T)
                # promoting non-string to string column
                copyto!(tltapes[col], 1, tlposlens[col], 1, tlrows)
                unset!(tlposlens, col, tlrows, 1)
            elseif TL == MISSINGTYPE && T == (INT | MISSING)
                fill!(tltapes[col], uint64(intsentinels[col]))
            elseif TL == MISSINGTYPE && pooled(T)
                fill!(tltapes[col], 0)
            elseif TL == MISSINGTYPE && missingtype(T)
                fill!(tltapes[col], sentinelvalue(TYPECODES[T & ~MISSING]))
            elseif TL == INT && (T == FLOAT || T == (FLOAT | MISSING))
                tape = tltapes[col]
                for j = 1:tlrows
                    @inbounds tape[j] = uint64(Float64(int64(tape[j])))
                end
            elseif TL == (INT | MISSING) && T == (FLOAT | MISSING)
                tape = tltapes[col]
                intsent = intsentinels[col]
                for j = 1:tlrows
                    @inbounds z = int64(tape[j])
                    @inbounds tape[j] = ifelse(z == intsent, sentinelvalue(Float64), uint64(Float64(z)))
                end
            elseif TL == (INT | MISSING) && T == (INT | MISSING)
                # synchronize int sentinel if needed
                if perthreadintsentinels[i][col] != intsentinels[col]
                    tape = tltapes[col]
                    newintsent = intsentinels[col]
                    oldintsent = perthreadintsentinels[i][col]
                    for j = 1:tlrows
                        @inbounds z = tape[j]
                        if z == oldintsent
                            @inbounds tape[j] = newintsent
                        end
                    end
                end
            elseif pooled(T)
                # synchronize pooled refs from each thread
                if !isassigned(refs, col)
                    refs[col] = tlrefs[col]
                    lastrefs[col] = tllastrefs[col]
                else
                    refrecodes = collect(UInt64(0):tllastrefs[col])
                    colrefs = refs[col]
                    recode = false
                    for (k, v) in tlrefs[col]
                        refvalue = get(colrefs, k, UInt64(0))
                        if refvalue != v
                            recode = true
                            if refvalue == 0
                                refvalue = (lastrefs[col] += UInt64(1))
                            end
                            colrefs[k] = refvalue
                            refrecodes[v + 1] = refvalue
                        end
                    end
                    if recode
                        tape = tltapes[col]
                        for j = 1:tlrows
                            tape[j] = refrecodes[tape[j] + 1]
                        end
                    end
                end
            end
        end
    end
    return perthreadrows, perthreadtapes, refs, typecodes, intsentinels
end

function parsetape(TR::Val{transpose}, ncols, typemap, tapes, poslens, buf, pos, len, limit, positions, pool, refs, lastrefs, rowsguess, types, debug, options::Parsers.Options{ignorerepeated}, coloptions) where {transpose, ignorerepeated}
    row = 0
    startpos = pos
    if pos <= len && len > 0
        while row < limit
            row += 1
            pos = parserow(row, TR, ncols, typemap, tapes, poslens, buf, pos, len, limit, positions, pool, refs, lastrefs, rowsguess, types, debug, options, coloptions)
            pos > len && break
            # if our initial row estimate was too few, we need to reallocate our tapes/poslens to read the rest of the file
            if row + 1 > rowsguess
                # (bytes left in file) / (avg bytes per row) == estimated rows left in file (+ 10 for kicks)
                estimated_rows_left = ceil(Int64, (len - pos) / ((pos - startpos) / row) + 10.0)
                newrowsguess = rowsguess + estimated_rows_left
                debug && reallocatetape(row, rowsguess, newrowsguess)
                for i = 1:ncols
                    resize!(tapes[i], newrowsguess)
                    if isassigned(poslens, i)
                        resize!(poslens[i], newrowsguess)
                    end
                end
                rowsguess = newrowsguess
            end
        end
    end
    return row, tapes, poslens
end

@noinline reallocatetape(row, old, new) = println("thread = $(Threads.threadid()) warning: didn't pre-allocate enough tape while parsing on row $row, re-allocating from $old to $new...")
@noinline notenoughcolumns(cols, ncols, row) = println("thread = $(Threads.threadid()) warning: only found $cols / $ncols columns on data row: $row. Filling remaining columns with `missing`")
@noinline toomanycolumns(cols, row) = println("thread = $(Threads.threadid()) warning: parsed expected $cols columns, but didn't reach end of line on data row: $row. Ignoring any extra columns on this row")
@noinline stricterror(T, buf, pos, len, code, row, col) = throw(Error("thread = $(Threads.threadid()) error parsing $T on row = $row, col = $col: \"$(String(buf[pos:pos+len-1]))\", error=$(Parsers.codes(code))"))
@noinline warning(T, buf, pos, len, code, row, col) = println("thread = $(Threads.threadid()) warning: error parsing $T on row = $row, col = $col: \"$(String(buf[pos:pos+len-1]))\", error=$(Parsers.codes(code))")
@noinline fatalerror(buf, pos, len, code, row, col) = throw(Error("thread = $(Threads.threadid()) fatal error, encountered an invalidly quoted field while parsing on row = $row, col = $col: \"$(String(buf[pos:pos+len-1]))\", error=$(Parsers.codes(code)), check your `quotechar` arguments or manually fix the field in the file itself"))

@inline function parserow(row, ::Val{transpose}, ncols, typemap, tapes, poslens, buf, pos, len, limit, positions, pool, refs, lastrefs, rowsguess, types, debug, options::Parsers.Options{ignorerepeated}, coloptions) where {transpose, ignorerepeated}
    for col = 1:ncols
        if transpose
            @inbounds pos = positions[col]
        end
        @inbounds T = types[col]
        @inbounds tape = tapes[col]
        @inbounds opts = coloptions === nothing ? options : coloptions[col]
        # @show typeof(tape), ArrayType{Int64}
        if tape isa SingleValueVector{Union{User, Missing}}
            pos, code = parsemissing!(buf, pos, len, opts, row, col)
        elseif tape isa SingleValueVector{Empty} || tape isa SingleValueVector{MissingType}
            pos, code = detect(tapes, buf, pos, len, opts, row, col, typemap, pool, refs, lastrefs, debug, types, poslens, rowsguess)
        elseif tape isa ArrayType{Int64}
            pos, code = parseint!(Union{User, Int64}, tape, tapes, buf, pos, len, opts, row, col, types, poslens)
        elseif tape isa ArrayType{Float64}
            pos, code = parsevalue!(Float64, T, tape, tapes, buf, pos, len, opts, row, col, types, poslens)
        elseif tape isa ArrayType{Date}
            pos, code = parsevalue!(Date, T, tape, tapes, buf, pos, len, opts, row, col, types, poslens)
        elseif tape isa ArrayType{DateTime}
            pos, code = parsevalue!(DateTime, T, tape, tapes, buf, pos, len, opts, row, col, types, poslens)
        elseif tape isa ArrayType{Time}
            pos, code = parsevalue!(Time, T, tape, tapes, buf, pos, len, opts, row, col, types, poslens)
        elseif tape isa ArrayType{Bool}
            pos, code = parsevalue!(Bool, T, tape, tapes, buf, pos, len, opts, row, col, types, poslens)
        elseif tape isa Vector{UInt32}
            pos, code = parsepooled!(T, tape, tapes, buf, pos, len, opts, row, col, rowsguess, pool, refs, lastrefs, types, poslens)
        elseif tape isa SentinelArray{PosLen,1,PosLen,Missing,Array{PosLen,1}}
            pos, code = parsestring!(T, tape, buf, pos, len, opts, row, col, types)
        # TODO: support all other integer types, float16, float32
        # in an else clause, we'll parse a string and call
        # Parsers.parse(T, str)
        end
        if transpose
            @inbounds positions[col] = pos
        else
            if col < ncols
                if Parsers.newline(code) || pos > len
                    options.silencewarnings || notenoughcolumns(col, ncols, row)
                    for j = (col + 1):ncols
                        # put in dummy missing values on the tape for missing columns
                        @inbounds tape = tapes[j]
                        tape[row] = missing
                        if isassigned(poslens, j)
                            setposlen!(poslens[j], row, Parsers.SENTINEL, pos, UInt64(0))
                        end
                        if !missingtype(T)
                            types[j] = Union{T, Missing}
                        end
                    end
                    break # from for col = 1:ncols
                end
            else
                if pos <= len && !Parsers.newline(code)
                    options.silencewarnings || toomanycolumns(ncols, row)
                    # ignore the rest of the line
                    pos = skiptorow(buf, pos, len, options.oq, options.e, options.cq, 1, 2)
                end
            end
        end
    end
    return pos
end

@inline function setposlen!(tape, row, code, pos, len)
    pos = Core.bitcast(UInt64, pos) << 20
    pos |= ifelse(Parsers.sentinel(code), MISSING_BIT, UInt64(0))
    pos |= ifelse(Parsers.escapedstring(code), ESCAPE_BIT, UInt64(0))
    @inbounds tape[row] = Core.bitcast(PosLen, pos | Core.bitcast(UInt64, len))
    return
end

function detect(tapes, buf, pos, len, options, row, col, typemap, pool, refs, lastrefs, debug, types, poslens, rowsguess)
    int, code, vpos, vlen, tlen = Parsers.xparse(Int64, buf, pos, len, options)
    if Parsers.invalidquotedfield(code)
        fatalerror(buf, pos, tlen, code, row, col)
    end
    if Parsers.sentinel(code) && code > 0
        @inbounds setposlen!(poslens[col], row, code, vpos, vlen)
        if types[col] == Empty
            @inbounds types[col] = MissingType
        end
        # return; parsing will continue to detect until a non-missing value is parsed
        @goto finaldone
    end
    if Parsers.ok(code) && !haskey(typemap, Int64)
        @inbounds setposlen!(poslens[col], row, code, vpos, vlen)
        tape = allocate(Int64, rowsguess)
        tape[row] = int
        tapes[col] = tape
        newT = Int64
        @goto done
    end
    float, code, vpos, vlen, tlen = Parsers.xparse(Float64, buf, pos, len, options)
    if Parsers.ok(code) && !haskey(typemap, Float64)
        @inbounds setposlen!(poslens[col], row, code, vpos, vlen)
        tape = allocate(Float64, rowsguess)
        tape[row] = float
        tapes[col] = tape
        newT = Float64
        @goto done
    end
    if options.dateformat === nothing
        date, code, vpos, vlen, tlen = Parsers.xparse(Date, buf, pos, len, options)
        if Parsers.ok(code) && !haskey(typemap, Date)
            @inbounds setposlen!(poslens[col], row, code, vpos, vlen)
            tape = allocate(Date, rowsguess)
            tape[row] = date
            tapes[col] = tape
            newT = Date
            @goto done
        end
        datetime, code, vpos, vlen, tlen = Parsers.xparse(DateTime, buf, pos, len, options)
        if Parsers.ok(code) && !haskey(typemap, DateTime)
            @inbounds setposlen!(poslens[col], row, code, vpos, vlen)
            tape = allocate(DateTime, rowsguess)
            tape[row] = datetime
            tapes[col] = tape
            newT = DateTime
            @goto done
        end
        time, code, vpos, vlen, tlen = Parsers.xparse(Time, buf, pos, len, options)
        if Parsers.ok(code) && !haskey(typemap, Time)
            @inbounds setposlen!(poslens[col], row, code, vpos, vlen)
            tape = allocate(Time, rowsguess)
            tape[row] = time
            tapes[col] = tape
            newT = Time
            @goto done
        end
    else
        # use user-provided dateformat
        DT = timetype(options.dateformat)
        dt, code, vpos, vlen, tlen = Parsers.xparse(DT, buf, pos, len, options)
        if Parsers.ok(code)
            @inbounds setposlen!(poslens[col], row, code, vpos, vlen)
            tape = allocate(DT, rowsguess)
            tape[row] = dt
            tapes[col] = tape
            newT = DT
            @goto done
        end
    end
    bool, code, vpos, vlen, tlen = Parsers.xparse(Bool, buf, pos, len, options)
    if Parsers.ok(code) && !haskey(typemap, Bool)
        @inbounds setposlen!(poslens[col], row, code, vpos, vlen)
        tape = allocate(Bool, rowsguess)
        tape[row] = bool
        tapes[col] = tape
        newT = Bool
        @goto done
    end
    _, code, vpos, vlen, tlen = Parsers.xparse(String, buf, pos, len, options)
    if pool > 0.0
        r = Dict{String, UInt32}()
        @inbounds refs[col] = r
        ref = getref!(r, PointerString(pointer(buf, vpos), vlen), lastrefs, col, code, options)
        @inbounds setposlen!(poslens[col], row, code, vpos, vlen)
        tape = zeros(UInt32, rowsguess)
        @inbounds tape[row] = ref
        tapes[col] = tape
        newT = PooledString
    else
        tape = allocate(Str, rowsguess)
        if Str === String
            @inbounds tape[row] = str(Str, buf, vpos, vlen)
        else
            setposlen!(tape, row, code, vpos, vlen)
        end
        tapes[col] = tape
        newT = String
        unset!(poslens, col, row, 1)
    end
@label done
    # if we're here, that means we found a non-missing value, so we need to update types
    if types[col] == MissingType
        @inbounds types[col] = Union{newT, Missing}
    else
        @inbounds types[col] = newT
    end
@label finaldone
    return pos + tlen, code
end

function parseint!(T, tape, tapes, buf, pos, len, options, row, col, types, poslens)
    x, code, vpos, vlen, tlen = Parsers.xparse(Int64, buf, pos, len, options)
    if code > 0
        if !Parsers.sentinel(code)
            @inbounds tape[row] = x
        else
            @inbounds types[col] = Union{T, Missing}
        end
        if !user(T)
            @inbounds setposlen!(poslens[col], row, code, vpos, vlen)
        end
    else
        if Parsers.invalidquotedfield(code)
            # this usually means parsing is borked because of an invalidly quoted field, hard error
            fatalerror(buf, pos, tlen, code, row, col)
        end
        if user(T)
            if !options.strict
                options.silencewarnings || warning(Int64, buf, pos, tlen, code, row, col)
                @inbounds types[col] = Union{T, Missing}
            else
                stricterror(Int64, buf, pos, tlen, code, row, col)
            end
        else
            # encountered an error parsing an Intger in auto-detect mode, so let's try Float64
            y, code, vpos, vlen, tlen = Parsers.xparse(Float64, buf, pos, len, options)
            if code > 0
                @inbounds tapes[col] = convert(Vector{Union{Float64, Missing}}, tape)
                @inbounds tapes[col][row] = y
                @inbounds types[col] = missingtype(T) ? Union{Float64, Missing} : Float64
                @inbounds setposlen!(poslens[col], row, code, vpos, vlen)
            else
                _, code, vpos, vlen, tlen = Parsers.xparse(String, buf, pos, len, options)
                tape = Vector{Union{Str, Missing}}(undef, length(tape))
                poslen = poslens[col]
                for i = 1:(row - 1)
                    @inbounds tape[i] = str(Str, buf, poslen[i])
                end
                unset!(poslens, col, row, 2)
                @inbounds types[col] = missingtype(T) ? Union{String, Missing} : String
            end
        end
    end
    return pos + tlen, code
end

function parsevalue!(::Type{type}, T, tape, tapes, buf, pos, len, options, row, col, types, poslens) where {type}
    x, code, vpos, vlen, tlen = Parsers.xparse(type, buf, pos, len, options)
    if code > 0
        if !Parsers.sentinel(code)
            @inbounds tape[row] = x
        else
            @inbounds types[col] = Union{type, Missing}
        end
        if !user(T)
            @inbounds setposlen!(poslens[col], row, code, vpos, vlen)
        end
    else
        if Parsers.invalidquotedfield(code)
            # this usually means parsing is borked because of an invalidly quoted field, hard error
            fatalerror(buf, pos, tlen, code, row, col)
        end
        if user(T)
            if !options.strict
                # code |= Parsers.SENTINEL # what was this for??
                options.silencewarnings || warning(type, buf, pos, tlen, code, row, col)
                @inbounds types[col] = Union{T, Missing}
            else
                stricterror(type, buf, pos, tlen, code, row, col)
            end
        else
            _, code, vpos, vlen, tlen = Parsers.xparse(String, buf, pos, len, options)
            tape = Vector{Union{Str, Missing}}(undef, length(tape))
            poslen = poslens[col]
            for i = 1:(row - 1)
                @inbounds tape[i] = str(Str, buf, poslen[i])
            end
            unset!(poslens, col, row, 2)
            @inbounds types[col] = missingtype(T) ? Union{String, Missing} : String
        end
    end
    return pos + tlen, code
end

@inline function parsestring!(T, tape, buf, pos, len, options, row, col, types)
    x, code, vpos, vlen, tlen = Parsers.xparse(String, buf, pos, len, options)
    if Str === String
        @inbounds tape[row] = str(Str, buf, vpos, vlen)
    else
        setposlen!(tape, row, code, vpos, vlen)
    end
    # setposlen!(tape, row, code, vpos, vlen)
    if Parsers.invalidquotedfield(code)
        # this usually means parsing is borked because of an invalidly quoted field, hard error
        fatalerror(buf, pos, tlen, code, row, col)
    end
    if Parsers.sentinel(code)
        @inbounds types[col] = Union{String, Missing}
    end
    return pos + tlen, code
end

function parsemissing!(buf, pos, len, options, row, col)
    x, code, vpos, vlen, tlen = Parsers.xparse(String, buf, pos, len, options)
    if Parsers.invalidquotedfield(code)
        # this usually means parsing is borked because of an invalidly quoted field, hard error
        fatalerror(buf, pos, tlen, code, row, col)
    end
    return pos + tlen, code
end

function getref!(x::Dict, key::PointerString, lastrefs, col, code, options)
    if Parsers.escapedstring(code)
        key2 = unescape(key, options.e)
        index = Base.ht_keyindex2!(x, key2)
    else
        index = Base.ht_keyindex2!(x, key)
    end
    if index > 0
        @inbounds found_key = x.vals[index]
        return found_key::UInt32
    else
        @inbounds new = (lastrefs[col] += UInt32(1))
        @inbounds Base._setindex!(x, new, Parsers.escapedstring(code) ? key2 : String(key), -index)
        return new
    end
end

function parsepooled!(T, tape, tapes, buf, pos, len, options, row, col, rowsguess, pool, refs, lastrefs, types, poslens)
    x, code, vpos, vlen, tlen = Parsers.xparse(String, buf, pos, len, options)
    if Parsers.invalidquotedfield(code)
        # this usually means parsing is borked because of an invalidly quoted field, hard error
        fatalerror(buf, pos, tlen, code, row, col)
    end
    if !isassigned(refs, col)
        r = Dict{String, UInt32}()
        @inbounds refs[col] = r
    else
        @inbounds r = refs[col]
    end
    if Parsers.sentinel(code)
        @inbounds types[col] = Union{T, Missing}
        ref = UInt32(0)
    else
        ref = getref!(r, PointerString(pointer(buf, vpos), vlen), lastrefs, col, code, options)
    end
    if !user(T) && (length(refs[col]) / rowsguess) > pool
        # promote to string
        tape = Vector{Union{Str, Missing}}(undef, length(tape))
        poslen = poslens[col]
        for i = 1:(row - 1)
            @inbounds tape[i] = str(Str, buf, poslen[i])
        end
        unset!(poslens, col, row, 2)
        @inbounds types[col] = missingtype(T) ? Union{String, Missing} : String
    else
        if !user(T)
            @inbounds setposlen!(poslens[col], row, code, vpos, vlen)
        end
        @inbounds tape[row] = ref
    end
    return pos + tlen, code
end
