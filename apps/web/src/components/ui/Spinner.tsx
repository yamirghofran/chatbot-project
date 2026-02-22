export function Spinner() {
  return (
    <>
      <style>{`
        @keyframes spinner-bar-fade {
          from { opacity: 1; }
          to   { opacity: 0.15; }
        }
      `}</style>
      <div style={{ position: "relative", width: 22, height: 22 }}>
        {Array.from({ length: 12 }, (_, i) => (
          <div
            key={i}
            style={{
              position: "absolute",
              left: "50%",
              top: "50%",
              width: 2,
              height: 6,
              marginLeft: -1,
              marginTop: -3,
              borderRadius: 2,
              backgroundColor: "currentColor",
              transform: `rotate(${i * 30}deg) translateY(-8px)`,
              animation: "spinner-bar-fade 1s linear infinite",
              animationDelay: `${-((12 - i) / 12).toFixed(4)}s`,
              opacity: 0.15,
            }}
          />
        ))}
      </div>
    </>
  );
}
