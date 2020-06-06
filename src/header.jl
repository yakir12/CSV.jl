struct Header{transpose, O, IO}
    name::String
    names::Vector{Symbol}
    rowsguess::Int64
    cols::Int
    e::UInt8
    buf::IO
    len::Int
    datapos::Int64
    options::O # Parsers.Options
    coloptions::Union{Nothing, Vector{Parsers.Options}}
    positions::Vector{Int64}
    types::Vector{Type}
    todrop::Vector{Int}
    pool::Float64
    categorical::Bool
end

"""
    isvaliddelim(delim)

Whether a character or string is valid for use as a delimiter.
"""
isvaliddelim(delim) = false
isvaliddelim(delim::Char) = delim != '\r' && delim != '\n' && delim != '\0'
isvaliddelim(delim::AbstractString) = all(isvaliddelim, delim)

"""
    checkvaliddelim(delim)

Checks whether a character or string is valid for use as a delimiter.  If
`delim` is `nothing`, it is assumed that the delimiter will be auto-selected.
Throws an error if `delim` is invalid.
"""
function checkvaliddelim(delim)
    delim != nothing && !isvaliddelim(delim) &&
        throw(ArgumentError("invalid delim argument = '$(escape_string(string(delim)))', "*
                            "the following delimiters are invalid: '\\r', '\\n', '\\0'"))
end

@inline function Header(source,
    # file options
    # header can be a row number, range of rows, or actual string vector
    header,
    normalizenames,
    datarow,
    skipto,
    footerskip,
    limit,
    transpose,
    comment,
    use_mmap,
    ignoreemptylines,
    threaded,
    select,
    drop,
    # parsing options
    missingstrings,
    missingstring,
    delim,
    ignorerepeated,
    quotechar,
    openquotechar,
    closequotechar,
    escapechar,
    dateformat,
    dateformats,
    decimal,
    truestrings,
    falsestrings,
    # type options
    type,
    types,
    typemap,
    categorical,
    pool,
    strict,
    silencewarnings,
    debug,
    parsingdebug,
    streaming)

    # initial argument validation and adjustment
    !isa(source, IO) && !isa(source, Vector{UInt8}) && !isa(source, Cmd) && !isfile(source) &&
        throw(ArgumentError("\"$source\" is not a valid file"))
    (types !== nothing && any(x->!isconcretetype(x) && !(x isa Union), types isa AbstractDict ? values(types) : types)) && throw(ArgumentError("Non-concrete types passed in `types` keyword argument, please provide concrete types for columns: $types"))
    if type !== nothing && standardize(type) == Empty
        throw(ArgumentError("$type isn't supported in the `type` keyword argument; must be one of: `Int64`, `Float64`, `Date`, `DateTime`, `Bool`, `Missing`, `PooledString`, `CategoricalString{UInt32}`, or `String`"))
    elseif types !== nothing && any(x->typecode(x) == Empty, types isa AbstractDict ? values(types) : types)
        T = nothing
        for x in (types isa AbstractDict ? values(types) : types)
            if typecode(x) == Empty
                T = x
                break
            end
        end
        throw(ArgumentError("unsupported type $T in the `types` keyword argument; must be one of: `Int64`, `Float64`, `Date`, `DateTime`, `Bool`, `Missing`, `PooledString`, `CategoricalString{UInt32}`, or `String`"))
    end
    checkvaliddelim(delim)
    ignorerepeated && delim === nothing && throw(ArgumentError("auto-delimiter detection not supported when `ignorerepeated=true`; please provide delimiter like `delim=','`"))
    if !(categorical isa Bool)
        @warn "categorical=$categorical is deprecated in favor of `pool=$categorical`; categorical is only used to determine CategoricalArray vs. PooledArrays"
        pool = categorical
        categorical = categorical > 0.0
    elseif categorical === true
        pool = categorical
    end
    header = (isa(header, Integer) && header == 1 && (datarow == 1 || skipto == 1)) ? -1 : header
    isa(header, Integer) && datarow != -1 && (datarow > header || throw(ArgumentError("data row ($datarow) must come after header row ($header)")))
    datarow = skipto !== nothing ? skipto : (datarow == -1 ? (isa(header, Vector{Symbol}) || isa(header, Vector{String}) ? 0 : last(header)) + 1 : datarow) # by default, data starts on line after header
    debug && println("header is: $header, datarow computed as: $datarow")
    # getsource will turn any input into a `Vector{UInt8}`
    buf = getsource(source, use_mmap)
    len = length(buf)
    # skip over initial BOM character, if present
    pos = consumeBOM(buf)

    oq = something(openquotechar, quotechar) % UInt8
    eq = escapechar % UInt8
    cq = something(closequotechar, quotechar) % UInt8
    trues = truestrings === nothing ? nothing : truestrings
    falses = falsestrings === nothing ? nothing : falsestrings
    sentinel = ((isempty(missingstrings) && missingstring == "") || (length(missingstrings) == 1 && missingstrings[1] == "")) ? missing : isempty(missingstrings) ? [missingstring] : missingstrings

    if delim === nothing
        del = isa(source, AbstractString) && endswith(source, ".tsv") ? UInt8('\t') :
            isa(source, AbstractString) && endswith(source, ".wsv") ? UInt8(' ') :
            UInt8('\n')
    else
        del = (delim isa Char && isascii(delim)) ? delim % UInt8 :
            (sizeof(delim) == 1 && isascii(delim)) ? delim[1] % UInt8 : delim
    end
    cmt = comment === nothing ? nothing : (pointer(comment), sizeof(comment))

    if footerskip > 0 && len > 0
        lastbyte = buf[end]
        endpos = (lastbyte == UInt8('\r') || lastbyte == UInt8('\n')) +
            (lastbyte == UInt8('\n') && buf[end - 1] == UInt8('\r'))
        revlen = skiptorow(ReversedBuf(buf), 1 + endpos, len, oq, eq, cq, 0, footerskip) - 2
        len -= revlen
        debug && println("adjusted for footerskip, len = $(len + revlen - 1) => $len")
    end

    if !transpose
        # step 1: detect the byte position where the column names start (headerpos)
        # and where the first data row starts (datapos)
        headerpos, datapos = detectheaderdatapos(buf, pos, len, oq, eq, cq, cmt, ignoreemptylines, header, datarow)
        debug && println("headerpos = $headerpos, datapos = $datapos")

        # step 2: detect delimiter (or use given) and detect number of (estimated) rows and columns
        d, rowsguess = detectdelimandguessrows(buf, headerpos, datapos, len, oq, eq, cq, del, cmt, ignoreemptylines)
        debug && println("estimated rows: $rowsguess")
        debug && println("detected delimiter: \"$(escape_string(d isa UInt8 ? string(Char(d)) : d))\"")

        # step 3: build Parsers.Options w/ parsing arguments
        wh1 = d == UInt(' ') ? 0x00 : UInt8(' ')
        wh2 = d == UInt8('\t') ? 0x00 : UInt8('\t')
        options = Parsers.Options(sentinel, wh1, wh2, oq, cq, eq, d, decimal, trues, falses, dateformat, ignorerepeated, ignoreemptylines, comment, true, parsingdebug, strict, silencewarnings)

        # step 4a: if we're ignoring repeated delimiters, then we ignore any
        # that start a row, so we need to check if we need to adjust our headerpos/datapos
        if ignorerepeated
            if headerpos > 0
                headerpos = Parsers.checkdelim!(buf, headerpos, len, options)
            end
            datapos = Parsers.checkdelim!(buf, datapos, len, options)
        end

        # step 4b: generate or parse column names
        names = detectcolumnnames(buf, headerpos, datapos, len, options, header, normalizenames)
        ncols = length(names)
        positions = Int64[]
    else
        # transpose
        d, rowsguess = detectdelimandguessrows(buf, pos, pos, len, oq, eq, cq, del, cmt, ignoreemptylines)
        wh1 = d == UInt(' ') ? 0x00 : UInt8(' ')
        wh2 = d == UInt8('\t') ? 0x00 : UInt8('\t')
        options = Parsers.Options(sentinel, wh1, wh2, oq, cq, eq, d, decimal, trues, falses, dateformat, ignorerepeated, ignoreemptylines, comment, true, parsingdebug, strict, silencewarnings)
        rowsguess, names, positions = detecttranspose(buf, pos, len, options, header, datarow, normalizenames)
        ncols = length(names)
        datapos = isempty(positions) ? 0 : positions[1]
    end
    debug && println("column names detected: $names")
    debug && println("byte position of data computed at: $datapos")

    # generate column options if applicable
    if dateformats === nothing || isempty(dateformats)
        coloptions = nothing
    elseif dateformats isa AbstractDict{String}
        coloptions = [haskey(dateformats, string(nm)) ? Parsers.Options(sentinel, wh1, wh2, oq, cq, eq, d, decimal, trues, falses, dateformats[string(nm)], ignorerepeated, ignoreemptylines, comment, true, parsingdebug, strict, silencewarnings) : options for nm in names]
    elseif dateformats isa AbstractDict{Symbol}
        coloptions = [haskey(dateformats, nm) ? Parsers.Options(sentinel, wh1, wh2, oq, cq, eq, d, decimal, trues, falses, dateformats[nm], ignorerepeated, ignoreemptylines, comment, true, parsingdebug, strict, silencewarnings) : options for nm in names]
    elseif dateformats isa AbstractDict{Int}
        coloptions = [haskey(dateformats, i) ? Parsers.Options(sentinel, wh1, wh2, oq, cq, eq, d, decimal, trues, falses, dateformats[i], ignorerepeated, ignoreemptylines, comment, true, parsingdebug, strict, silencewarnings) : options for i = 1:ncols]
    end
    debug && println("column options generated as: $(something(coloptions, ""))")

    # deduce initial column types for parsing based on whether any user-provided types were provided or not
    T = type === nothing ? (streaming ? Union{String, Missing} : Empty) : Union{standardize(type), User}
    if types isa Vector
        types = Type[Union{standardize(T), User} for T in types]
        categorical = categorical | any(x->x == CategoricalString{UInt32}, types)
    elseif types isa AbstractDict
        types = initialtypes(T, types, names)
        categorical = categorical | any(x->x == CategoricalString{UInt32}, values(types))
    else
        types = Type[T for _ = 1:ncols]
    end
    if streaming
        for i = 1:ncols
            T = types[i]
            if pooled(T)
                @warn "pooled column types not allowed in `CSV.Rows` (column number = $i)"
                types[i] = T >: Missing ? Union{String, Missing} : String
            end
        end
    end
    # set any unselected columns to typecode USER | MISSING
    todrop = Int[]
    if select !== nothing && drop !== nothing
        error("`select` and `drop` keywords were both provided; only one or the other is allowed")
    elseif select !== nothing
        if select isa AbstractVector{Int}
            for i = 1:ncols
                i in select || push!(todrop, i)
            end
        elseif select isa AbstractVector{Symbol} || select isa AbstractVector{<:AbstractString}
            select = map(Symbol, select)
            for i = 1:ncols
                names[i] in select || push!(todrop, i)
            end
        elseif select isa AbstractVector{Bool}
            for i = 1:ncols
                select[i] || push!(todrop, i)
            end
        elseif select isa Base.Callable
            for i = 1:ncols
                select(i, names[i]) || push!(todrop, i)
            end
        else
            error("`select` keyword argument must be an `AbstractVector` of `Int`, `Symbol`, `String`, or `Bool`, or a selector function of the form `(i, name) -> keep::Bool`")
        end
    elseif drop !== nothing
        if drop isa AbstractVector{Int}
            for i = 1:ncols
                i in drop && push!(todrop, i)
            end
        elseif drop isa AbstractVector{Symbol} || drop isa AbstractVector{<:AbstractString}
            drop = map(Symbol, drop)
            for i = 1:ncols
                names[i] in drop && push!(todrop, i)
            end
        elseif drop isa AbstractVector{Bool}
            for i = 1:ncols
                drop[i] && push!(todrop, i)
            end
        elseif drop isa Base.Callable
            for i = 1:ncols
                drop(i, names[i]) && push!(todrop, i)
            end
        else
            error("`drop` keyword argument must be an `AbstractVector` of `Int`, `Symbol`, `String`, or `Bool`, or a selector function of the form `(i, name) -> keep::Bool`")
        end
    end
    for i in todrop
        types[i] = Union{User, Missing}
    end
    debug && println("computed types are: $types")
    pool = pool === true ? 1.0 : pool isa Float64 ? pool : 0.0
    return Header{transpose, typeof(options), typeof(buf)}(
        getname(source),
        names,
        rowsguess,
        ncols,
        eq,
        buf,
        len,
        datapos,
        options,
        coloptions,
        positions,
        types,
        todrop,
        pool,
        categorical
    )
end
