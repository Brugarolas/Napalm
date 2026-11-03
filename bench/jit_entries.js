class EntryBench {
    static kernel(x) {
        let a = x * 1.0001 + 0.25
        let b = a * a - a / 3.0
        let c = b * 0.75 + a * 0.5
        let d = c * c / (b + 1.0)
        return d + b - a + c / 7.0
    }

    static run(count) {
        let total = 0
        let i = 0
        while (i < count) {
            total = total + this.kernel(i % 97 + 0.125)
            i = i + 1
        }
        return total
    }
}

EntryBench.run(1000)
var t0 = performance.now();     // 1) start
let checksum = EntryBench.run(1000000)
var t1 = performance.now();     // 2) end
var elapsed = t1 - t0
var secs = elapsed / 1000;           // 3) milliseconds to seconds
console.log(`checksum: ${checksum}`)
console.log(`Execution time: ${secs.toFixed(6)} s`);  // 4) print