import { guardLabel, ms, pct, similarity } from "./format";

describe("format", () => {
  it("formats percentages", () => {
    expect(pct(0.632)).toBe("63.2%");
    expect(pct(0.009, 1)).toBe("0.9%");
    expect(pct(1, 0)).toBe("100%");
  });

  it("formats latency in ms, switching to seconds", () => {
    expect(ms(13.71)).toBe("13.7 ms");
    expect(ms(1240)).toBe("1.24 s");
  });

  it("formats similarity to two decimals", () => {
    expect(similarity(0.7)).toBe("0.70");
  });

  it("labels known guards and humanises unknown ones", () => {
    expect(guardLabel("negation")).toBe("Negation differs");
    expect(guardLabel("some_new_guard")).toBe("some new guard");
  });
});
