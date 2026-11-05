#[test]
fn napalm_strings_search_positions_and_overlapping_matches() {
    run_napalm(r#"
        const text = 'aé😀aé😀';
        const result = [text.includes('é😀'), text.contains('é😀'), text.includes(''),
            text.includes('missing'), text.indexOf('é'), text.indexOf('é', 2),
            text.indexOf('é', -9), text.indexOf('é', 99), text.indexOf('', 99),
            text.lastIndexOf('é'), text.lastIndexOf('é', 3), text.lastIndexOf('é', -1),
            text.lastIndexOf('', 99), text.lastIndexOf('missing'),
            'aaa'.lastIndexOf('aa'), 'aaaa'.lastIndexOf('aa', 1),
            'ééé'.lastIndexOf('éé', 1), '😀😀😀'.lastIndexOf('😀😀', 0),
            'a😀é'.lastIndexOf('é', 0), ''.lastIndexOf(''), ''.indexOf('x')];
    "#, "[true, true, true, false, 1, 4, 1, -1, 6, 4, 1, -1, 6, -1, 1, 1, 1, 0, -1, 0, -1]");
}

#[test]
fn napalm_strings_subscripts_use_scalar_indices() {
    run_napalm(r#"
        const text: string = 'aé😀';
        const dynamic: any = text;
        const result = [text[0], text[1], text[2], text[-1], text[-3],
            text[3], text[-4], ''[0], dynamic[2], text[1 + 1], text];
    "#, "[a, é, 😀, 😀, a, null, null, null, 😀, 😀, aé😀]");
}

#[test]
fn napalm_strings_padding_and_repeat() {
    run_napalm(r#"
        const text = 'é😀';
        const result = [text.padStart(5, 'ab'), text.padEnd(5, '😀é'),
            'x'.padStart(3), 'x'.padEnd(3), text.padStart(1), text.padEnd(-1),
            text.padStart(9, ''), ''.padEnd(3, 'é😀'),
            text.repeat(3), text.repeat(0).length, ''.repeat(9007199254740991).length,
            text.repeat(1), text];
    "#, "[abaé😀, é😀😀é😀,   x, x  , é😀, é😀, é😀, é😀é, é😀é😀é😀, 0, 0, é😀, é😀]");
}

#[test]
fn napalm_strings_split_empty_separator_limits_and_gc() {
    run_napalm(r#"
        const text = ',a,,é😀,';
        const parts = text.split(',');
        System.gc();
        const result = [parts, text.split(',', 3), text.split(',', 0).length,
            'é😀'.split(''), 'é😀'.split('', 1), ''.split('').length,
            ''.split(',').length, ''.split().length, 'abc'.split(),
            'abc'.split('x'), 'a--b--'.split('--'), 'aaa'.split('aa'),
            'a.b'.split('.'), 'a\0b'.split('\0'), text];
    "#, "[[, a, , é😀, ], [, a, ], 0, [é, 😀], [é], 0, 1, 1, [abc], [abc], [a, b, ], [, a], [a, b], [a, b], ,a,,é😀,]");
}

#[test]
fn napalm_strings_replace_literal_first_all_and_empty_search() {
    run_napalm(r#"
        const text = 'é😀é😀';
        const result = [text.replace('é', 'x'), text.replaceAll('é', 'x'),
            text.replace('absent', 'x'), text.replaceAll('😀', ''),
            'aaaaa'.replaceAll('aa', 'x'), 'a.b.'.replaceAll('.', '$&'),
            'é😀'.replace('', '-'), 'é😀'.replaceAll('', '-'),
            ''.replace('', 'x'), ''.replaceAll('', 'x'),
            'a'.replaceAll('a', 'aa'), 'x'.replace('x', '$1$$'), text];
    "#, "[x😀é😀, x😀x😀, é😀é😀, éé, xxa, a$&b$&, -é😀, -é-😀-, x, x, aa, $1$$, é😀é😀]");
}

#[test]
fn napalm_strings_invalid_arguments_are_catchable() {
    run_napalm(r#"
        let errors = 0;
        try { 'a'.includes(1); } catch (e) { errors++; }
        try { 'a'.indexOf('a', 0.5); } catch (e) { errors++; }
        try { 'a'.lastIndexOf('a', Num.nan); } catch (e) { errors++; }
        try { 'a'[false]; } catch (e) { errors++; }
        try { 'a'[0.5]; } catch (e) { errors++; }
        try { 'a'.padStart(2.5); } catch (e) { errors++; }
        try { 'a'.padEnd(Num.infinity); } catch (e) { errors++; }
        try { 'a'.padStart(3, 1); } catch (e) { errors++; }
        try { 'a'.repeat(-1); } catch (e) { errors++; }
        try { 'a'.repeat(0.5); } catch (e) { errors++; }
        try { 'a'.repeat(Num.infinity); } catch (e) { errors++; }
        try { 'a'.repeat('2'); } catch (e) { errors++; }
        try { 'a'.split(1); } catch (e) { errors++; }
        try { 'a'.split('', -1); } catch (e) { errors++; }
        try { 'a'.split('', 1.5); } catch (e) { errors++; }
        try { 'a'.replace(1, 'b'); } catch (e) { errors++; }
        try { 'a'.replaceAll('a', null); } catch (e) { errors++; }
        try { 'abcd'.repeat(1e30); } catch (e) { errors++; }
        try { 'a'.padEnd(1e30, '😀'); } catch (e) { errors++; }
        const result = errors;
    "#, "19");
}

#[test]
fn napalm_strings_hot_dispatch_and_gc() {
    run_napalm_checked(r#"
        let total = 0;
        let parts = [];
        for (let i = 0; i < 100; i++) {
            const text = 'é😀'.repeat(2).padStart(5, '-').padEnd(6, '!');
            parts = text.replace('é', 'a').replaceAll('😀', 'b').split('b');
            if (text.includes('é') && text[2] == '😀') {
                total += text.indexOf('é', 2) + text.lastIndexOf('😀', 3) + parts.length;
            }
            if (i % 10 == 0) System.gc();
        }
        System.gc();
        const result = [total, parts];
    "#, "[800, [-a, é, !]]", true);
}
