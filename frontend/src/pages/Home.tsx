import { Box, Typography, Button, Chip, Stack } from "@mui/material"
import { useNavigate } from "react-router-dom"

function Home() {
  const navigate = useNavigate()

  const stats = [
    { val: "YOLOv8",  lbl: "Detección"  },
    { val: "EasyOCR", lbl: "Lectura OCR" },
    { val: "99ms",    lbl: "Promedio"    },
  ]

  return (
    <Box
      sx={{
        minHeight: "100vh",
        bgcolor: "#080c18",
        color: "slate",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* ── HEADER ── */}
      <Box sx={{ px: 4, py: 3, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Typography fontWeight={800} fontSize={16} color="#fff">
          Traffic<Box component="span" sx={{ color: "#22d3ee" }}>Vision</Box>
        </Typography>
        <Typography
          component="a"
          href="http://localhost:8000/docs"
          target="_blank"
          fontSize={12}
          sx={{ color: "#475569", textDecoration: "none", "&:hover": { color: "#cbd5e1" } }}
        >
          API docs ↗
        </Typography>
      </Box>

      {/* ── HERO ── */}
      <Box
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
          px: 3,
          gap: 3.5,
        }}
      >
        {/* Badge */}
        <Chip
          label="Control de Placas · Seguridad Vial"
          size="small"
          sx={{
            bgcolor: "rgba(34,211,238,0.06)",
            border: "1px solid rgba(34,211,238,0.2)",
            color: "#22d3ee",
            fontWeight: 700,
            fontSize: 10,
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            height: 28,
          }}
        />

        {/* Heading */}
        <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
          <Typography
            fontWeight={900}
            sx={{
              fontSize: { xs: "4.6rem", md: "3.8rem" },
              lineHeight: 1.05,
              letterSpacing: "-0.03em",
              color: "#fff",
            }}
          >
            Reconocimiento<br />
            de{" "}
            <Box component="span" sx={{ color: "#22d3ee" }}>Placas</Box>
          </Typography>
          <Typography sx={{ color: "#64748b", fontSize: 14, maxWidth: 380, mx: "auto", lineHeight: 1.7 }}>
            Detección y lectura automática de placas vehiculares usando YOLOv8 + EasyOCR.
          </Typography>
        </Box>

        {/* CTA */}
        <Button
          onClick={() => navigate("/read-plate")}
          variant="contained"
          sx={{
            bgcolor: "#22d3ee",
            color: "#080c18",
            fontWeight: 800,
            fontSize: 13,
            px: 4,
            py: 1.4,
            borderRadius: 3,
            textTransform: "none",
            boxShadow: "0 8px 24px rgba(34,211,238,0.2)",
            "&:hover": { bgcolor: "#38bdf8", boxShadow: "0 8px 32px rgba(34,211,238,0.35)" },
          }}
        >
          Analizar placa →
        </Button>

        {/* Stats */}
        <Stack direction="row" spacing={5} mt={1}>
          {stats.map((s) => (
            <Box key={s.lbl} sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 0.3 }}>
              <Typography fontWeight={900} fontSize={18} color="#fff">{s.val}</Typography>
              <Typography fontSize={10} sx={{ color: "#475569", textTransform: "uppercase", letterSpacing: "0.1em" }}>
                {s.lbl}
              </Typography>
            </Box>
          ))}
        </Stack>
      </Box>

      {/* ── FOOTER ── */}
      <Typography textAlign="center" pb={3} fontSize={11} sx={{ color: "#1e293b" }}>
        Backend ·{" "}
        <Box component="span" sx={{ fontFamily: "monospace", color: "#334155" }}>
          localhost:8000
        </Box>
      </Typography>
    </Box>
  )
}

export default Home