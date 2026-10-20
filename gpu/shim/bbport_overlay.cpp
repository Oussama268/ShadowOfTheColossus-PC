// SPDX-License-Identifier: GPL-2.0-or-later
#include "bbport_overlay.h"

#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <mutex>

#include <SDL3/SDL.h>
#include "bbport_settings.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "stb_image.h"
#include <unordered_map>
#include "imgui_impl_vulkan.h"
#include "video_core/renderer_vulkan/vk_framegen.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/sotc_fps_cap.h"
#include "video_core/sotc_ui_lang.h"
#include "video_core/sotc_stats.h"

// DejaVu Sans (Cyrillic), embedded (third_party/fonts, Bitstream Vera license).
#ifdef _WIN32
asm(".section .rdata,\"dr\"\n"
    ".balign 16\n"
    ".global bb_font_ttf\n"
    "bb_font_ttf:\n"
    ".incbin \"" BB_FONT_PATH "\"\n"
    ".global bb_font_ttf_end\n"
    "bb_font_ttf_end:\n"
    ".text\n");
#else
asm(".section .rodata\n"
    ".balign 16\n"
    ".hidden bb_font_ttf\n"
    ".global bb_font_ttf\n"
    "bb_font_ttf:\n"
    ".incbin \"" BB_FONT_PATH "\"\n"
    ".hidden bb_font_ttf_end\n"
    ".global bb_font_ttf_end\n"
    "bb_font_ttf_end:\n"
    ".previous\n");
#endif
extern "C" const unsigned char bb_font_ttf[];
extern "C" const unsigned char bb_font_ttf_end[];

namespace BbOverlay {

namespace {

std::mutex imgui_mutex; // the ImGui context: window thread (input) and present thread
bool initialized = false;
// The game's text dialog (SetTextEntry), guarded by imgui_mutex.
bool text_entry_active = false;
std::string text_entry_prompt, text_entry_text;
// The game's save/load slot list (OpenSaveList), guarded by imgui_mutex.
bool list_active = false;
std::string list_title;
std::vector<std::string> list_items;
// sotc: each save's picture (icon0.png of its sce_sys), shown to the left of its text as in the
// PS4 and shadPS4 lists. Loaded once per file, kept for the session.
std::vector<std::string> list_icons, pending_icons;
std::unordered_map<std::string, ImTextureData*> icon_textures;

ImTextureData* IconTexture(const std::string& path) {
    if (path.empty()) {
        return nullptr;
    }
    if (const auto it = icon_textures.find(path); it != icon_textures.end()) {
        return it->second;
    }
    ImTextureData* tex = nullptr;
    int w = 0, h = 0, n = 0;
    if (unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &n, 4)) {
        tex = IM_NEW(ImTextureData)();
        tex->Create(ImTextureFormat_RGBA32, w, h);
        std::memcpy(tex->GetPixels(), pixels, size_t(w) * h * 4);
        stbi_image_free(pixels);
        ImGui::RegisterUserTexture(tex);
    }
    icon_textures[path] = tex;
    return tex;
}
int list_selected = 0, list_result = -1;
int list_confirm = -1;  // item waiting for the delete confirmation, or -1
int list_first_deletable = 0;
bool list_scroll = false, list_stick_down = false, list_stick_up = false;
float base_scale = 1.0f;

// Present rate for the FPS counter.
std::chrono::steady_clock::time_point last_present{};
float frame_ms_avg = 0.0f;

// sotc: a short message at the top of the screen (F8: the frame generation mode): its text
// (under imgui_mutex) and until when, in steady clock milliseconds.
std::string toast_text;
std::atomic<long long> toast_until_ms{0};

long long NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

bool ToastShown() {
    return NowMs() < toast_until_ms.load(std::memory_order_relaxed);
}

// sotc: the performance overlay (show_fps=1 in sotc.ini, or SOTC_FPS / SOTC_OVERLAY=1; F10
// toggles it). History of present intervals (one per game flip), present thread only.
constexpr int kHistory = 240;
float frame_hist[kHistory]{};
int hist_pos = 0, hist_count = 0;
uint64_t session_frames = 0, session_late = 0, session_stutters = 0;
std::atomic<bool> overlay_on{[] {
    if (const char* v = std::getenv("SOTC_OVERLAY")) {
        return v[0] != '0';
    }
    return std::getenv("SOTC_FPS") != nullptr;
}()};

// Frame time the game aims for: 33.3 ms (30 FPS) or 16.7 ms (60 FPS), from the median.
float TargetMs() {
    if (hist_count < 8) {
        return 33.3f;
    }
    float sorted[kHistory];
    std::copy(frame_hist, frame_hist + hist_count, sorted);
    std::nth_element(sorted, sorted + hist_count / 2, sorted + hist_count);
    return sorted[hist_count / 2] < 25.0f ? 16.7f : 33.3f;
}

bool OverlayOn() {
    return BbSettings::Get().show_fps || overlay_on.load(std::memory_order_relaxed);
}

void RecordFrame(float ms) {
    frame_hist[hist_pos] = ms;
    hist_pos = (hist_pos + 1) % kHistory;
    hist_count = std::min(hist_count + 1, kHistory);
    ++session_frames;
    if (ms > TargetMs() * 1.1f) {
        ++session_late;
    }
    if (ms > 50.0f) {
        ++session_stutters;
    }
}

float PixelDensity(SDL_WindowID id) {
    SDL_Window* window = SDL_GetWindowFromID(id);
    const float density = window ? SDL_GetWindowPixelDensity(window) : 1.0f;
    return density > 0.0f ? density : 1.0f;
}

// The game's text dialog: what is typed, and how to finish (keyboard or controller).
void TextEntryBox() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                   viewport->WorkPos.y + viewport->WorkSize.y * 0.42f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(viewport->WorkSize.x * 0.32f, 0.0f),
                                        ImVec2(viewport->WorkSize.x * 0.8f, FLT_MAX));
    ImGui::SetNextWindowBgAlpha(0.92f);
    ImGui::Begin("##text_entry", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    ImGui::TextColored(ImVec4(0.85f, 0.72f, 0.45f, 1.0f), "%s", text_entry_prompt.c_str());
    ImGui::Separator();
    ImGui::SetWindowFontScale(1.4f);
    ImGui::Text("%s_", text_entry_text.c_str());
    ImGui::SetWindowFontScale(1.0f);
    ImGui::Separator();
    ImGui::TextDisabled("Keyboard: type, Backspace to erase, Enter = OK, Esc = cancel");
    ImGui::TextDisabled("Controller: Cross (A) = OK, Circle (B) = cancel");
    ImGui::End();
}

// The game's save/load slot list: one entry per save, the selected one highlighted.
void SaveListBox() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                   viewport->WorkPos.y + viewport->WorkSize.y * 0.5f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x * 0.46f, viewport->WorkSize.y * 0.62f),
                             ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.94f);
    ImGui::Begin("##save_list", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoSavedSettings);
    ImGui::SetWindowFontScale(1.25f);
    ImGui::PushTextWrapPos(0.0f); // sotc: save dialog messages are whole sentences
    ImGui::TextColored(ImVec4(0.85f, 0.72f, 0.45f, 1.0f), "%s", list_title.c_str());
    ImGui::PopTextWrapPos();
    ImGui::SetWindowFontScale(1.0f);
    ImGui::Separator();
    const float footer = ImGui::GetTextLineHeightWithSpacing() * 2.2f;
    ImGui::BeginChild("##items", ImVec2(0.0f, -footer));
    // sotc: rows with the save's picture on the left (16:9 of 228x128) and its text on the right.
    const bool any_icon = std::any_of(list_icons.begin(), list_icons.end(),
                                      [](const std::string& p) { return !p.empty(); });
    const float row_h = any_icon ? ImGui::GetTextLineHeightWithSpacing() * 4.2f : 0.0f;
    for (int i = 0; i < static_cast<int>(list_items.size()); ++i) {
        ImGui::PushID(i);
        const bool selected = i == list_selected;
        ImTextureData* icon = i < static_cast<int>(list_icons.size()) ? IconTexture(list_icons[i]) : nullptr;
        const ImVec2 start = ImGui::GetCursorPos();
        const bool clicked =
            any_icon ? ImGui::Selectable("##row", selected,
                                         ImGuiSelectableFlags_AllowDoubleClick |
                                             ImGuiSelectableFlags_AllowOverlap,
                                         ImVec2(0.0f, row_h))
                     : ImGui::Selectable(list_items[i].c_str(), selected,
                                         ImGuiSelectableFlags_AllowDoubleClick);
        if (clicked) {
            list_selected = i;
            if (ImGui::IsMouseDoubleClicked(0)) {
                list_result = i;
            }
        }
        if (any_icon) {
            const ImVec2 end = ImGui::GetCursorPos();
            const float pad = row_h * 0.06f;
            const float img_h = row_h - pad * 2.0f;
            const float img_w = icon ? img_h * float(icon->Width) / float(std::max(icon->Height, 1))
                                     : img_h * 228.0f / 128.0f;
            ImGui::SetCursorPos(ImVec2(start.x + pad, start.y + pad));
            if (icon) {
                ImGui::Image(icon->GetTexRef(), ImVec2(img_w, img_h));
            } else {
                ImGui::Dummy(ImVec2(img_w, img_h));
            }
            const float text_x = start.x + pad * 3.0f + img_w;
            const float text_h = ImGui::CalcTextSize(list_items[i].c_str()).y;
            ImGui::SetCursorPos(ImVec2(text_x, start.y + std::max(pad, (row_h - text_h) * 0.5f)));
            ImGui::TextUnformatted(list_items[i].c_str());
            ImGui::SetCursorPos(end);
        }
        if (selected && list_scroll) {
            ImGui::SetScrollHereY(0.5f);
        }
        ImGui::Separator();
        ImGui::PopID();
    }
    list_scroll = false;
    ImGui::EndChild();
    ImGui::Separator();
    if (list_confirm >= 0) {
        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f),
                           "Apagar este save?   Enter / X (A): sim    Esc / O (B): nao");
    } else if (list_first_deletable >= static_cast<int>(list_items.size())) { // sotc: YES/NO
        ImGui::TextDisabled("Setas / direcional: escolher    Enter / X (A): confirmar    "
                            "Esc / O (B): nao");
    } else {
        ImGui::TextDisabled("Setas / direcional: escolher    Enter / X (A): confirmar    "
                            "Esc / O (B): voltar    Del / Triangulo (Y): apagar");
    }
    ImGui::End();
}

// sotc: the port's loading screen while the game shows nothing (boot, shader cache, long loads):
// the title, a turning arc and "Loading…" in the launcher's language, over a dark backdrop.
void LoadingScreen() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowBgAlpha(0.88f);
    ImGui::Begin("##loading", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 size = viewport->WorkSize;
    const ImVec2 origin = viewport->WorkPos;
    const float unit = size.y / 1080.0f;
    const ImU32 gold = IM_COL32(214, 184, 120, 255), dim = IM_COL32(150, 150, 150, 255);
    const char* title = "SHADOW OF THE COLOSSUS";
    const float title_px = 46.0f * unit;
    const ImVec2 title_size = ImGui::GetFont()->CalcTextSizeA(title_px, FLT_MAX, 0.0f, title);
    draw->AddText(ImGui::GetFont(), title_px,
                  ImVec2(origin.x + (size.x - title_size.x) * 0.5f, origin.y + size.y * 0.42f),
                  gold, title);
    // Turning arc in the lower right corner, like a console's loading mark.
    const float t = float(NowMs() % 100000) / 1000.0f;
    const ImVec2 centre(origin.x + size.x - 90.0f * unit, origin.y + size.y - 90.0f * unit);
    const float radius = 26.0f * unit;
    draw->PathArcTo(centre, radius, t * 4.0f, t * 4.0f + 4.4f, 40);
    draw->PathStroke(gold, 0, 4.0f * unit);
    const char* text = SotcUi::T("Carregando…", "Loading…");
    const float text_px = 24.0f * unit;
    const ImVec2 text_size = ImGui::GetFont()->CalcTextSizeA(text_px, FLT_MAX, 0.0f, text);
    draw->AddText(ImGui::GetFont(), text_px,
                  ImVec2(centre.x - radius - 18.0f * unit - text_size.x, centre.y - text_size.y * 0.5f),
                  dim, text);
    // While the shader cache is read at boot: a bar with the percentage.
    const u32 total = SotcStats::warmup_total.load(std::memory_order_relaxed);
    if (total > 0) {
        const u32 done = std::min(SotcStats::warmup_done.load(std::memory_order_relaxed), total);
        const float frac = float(done) / float(total);
        const float bar_w = size.x * 0.4f, bar_h = 10.0f * unit;
        const ImVec2 bar0(origin.x + (size.x - bar_w) * 0.5f, origin.y + size.y * 0.42f + title_px + 40.0f * unit);
        draw->AddRectFilled(bar0, ImVec2(bar0.x + bar_w, bar0.y + bar_h), IM_COL32(60, 60, 64, 255), bar_h * 0.5f);
        draw->AddRectFilled(bar0, ImVec2(bar0.x + bar_w * frac, bar0.y + bar_h), gold, bar_h * 0.5f);
        char line[128];
        std::snprintf(line, sizeof(line), "%s %d%%  (%u / %u)", SotcUi::T("Carregando shaders…", "Loading shaders…"),
                      int(frac * 100.0f + 0.5f), done, total);
        const float line_px = 22.0f * unit;
        const ImVec2 line_size = ImGui::GetFont()->CalcTextSizeA(line_px, FLT_MAX, 0.0f, line);
        draw->AddText(ImGui::GetFont(), line_px,
                      ImVec2(origin.x + (size.x - line_size.x) * 0.5f, bar0.y + bar_h + 14.0f * unit), dim, line);
    }
    ImGui::End();
}

void FpsCounter() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float pad = 12.0f * base_scale;
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - pad,
                                   viewport->WorkPos.y + pad),
                            ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::Begin("##fps", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    // Latest frames in order, oldest first.
    float frames[kHistory];
    for (int i = 0; i < hist_count; ++i) {
        frames[i] = frame_hist[(hist_pos - hist_count + i + kHistory) % kHistory];
    }
    const float target = TargetMs();
    float sum = 0.0f, worst = 0.0f;
    int late = 0;
    for (int i = 0; i < hist_count; ++i) {
        sum += frames[i];
        worst = std::max(worst, frames[i]);
        late += frames[i] > target * 1.1f;
    }
    float sorted[kHistory];
    std::copy(frames, frames + hist_count, sorted);
    std::sort(sorted, sorted + hist_count);
    const float p99 = hist_count ? sorted[std::min(hist_count - 1, hist_count * 99 / 100)] : 0.0f;
    const float avg_fps = sum > 0.0f ? 1000.0f * hist_count / sum : 0.0f;
    const float seconds = sum / 1000.0f;

    const ImVec4 green(0.45f, 0.9f, 0.45f, 1.0f), yellow(1.0f, 0.82f, 0.3f, 1.0f),
        red(1.0f, 0.4f, 0.35f, 1.0f), grey(0.7f, 0.7f, 0.7f, 1.0f);
    const auto rate_color = [&](float ms) {
        return ms <= target * 1.1f ? green : ms <= target * 1.6f ? yellow : red;
    };
    const float now_ms = frame_ms_avg;
    ImGui::TextColored(rate_color(now_ms), "%.0f FPS", now_ms > 0.0f ? 1000.0f / now_ms : 0.0f);
    ImGui::SameLine();
    ImGui::TextColored(grey, SotcUi::T("%.1f ms   alvo %.0f FPS", "%.1f ms   target %.0f FPS"), now_ms, 1000.0f / target);

    // Frame time graph, 0 to 2.2x the target, with the target line.
    const float graph_max = target * 2.2f;
    const ImVec2 graph_size(260.0f * base_scale, 54.0f * base_scale);
    ImGui::PushStyleColor(ImGuiCol_PlotLines, late ? yellow : green);
    ImGui::PlotLines("##frametimes", frames, hist_count, 0, nullptr, 0.0f, graph_max, graph_size);
    ImGui::PopStyleColor();
    const ImVec2 lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
    const float y = hi.y - (hi.y - lo.y) * (target / graph_max);
    ImGui::GetWindowDrawList()->AddLine(ImVec2(lo.x, y), ImVec2(hi.x, y),
                                        IM_COL32(120, 200, 255, 160), 1.0f);

    ImGui::Text(SotcUi::T("%.0f s: média %.1f | 1%% low %.1f | pior %.0f ms", "%.0f s: avg %.1f | 1%% low %.1f | worst %.0f ms"), seconds, avg_fps,
                p99 > 0.0f ? 1000.0f / p99 : 0.0f, worst);
    ImGui::TextColored(late ? (late > 3 ? red : yellow) : green, SotcUi::T("Atrasados: %d", "Late: %d"), late);
    ImGui::SameLine();
    ImGui::TextColored(grey, SotcUi::T("| sessão %llu (%.1f%%), travadas >50 ms %llu", "| session %llu (%.1f%%), stutters >50 ms %llu"),
                       static_cast<unsigned long long>(session_late),
                       session_frames ? 100.0 * session_late / session_frames : 0.0,
                       static_cast<unsigned long long>(session_stutters));
    if (SotcStats::kFpsLog) {
        ImGui::TextColored(grey, "CPU %.0f cmds/image | relectures GPU %.0f/s (%.0f ms/s)",
                           SotcStats::ov_cmds_per_frame.load(), SotcStats::ov_readbacks_per_s.load(),
                           SotcStats::ov_readback_ms_per_s.load());
        const int cause = SotcStats::ov_worst_cause.load();
        ImGui::TextColored(grey, "Shaders compiles %u (5 s), %llu session",
                           SotcStats::ov_compiles_window.load(),
                           static_cast<unsigned long long>(SotcStats::ov_compiles_total.load()));
        if (cause >= 0) {
            ImGui::SameLine();
            ImGui::TextColored(grey, SotcUi::T("| pior quadro: %s %.0f ms", "| worst frame: %s %.0f ms"),
                               SotcStats::kFrameCauseNames[cause],
                               SotcStats::ov_worst_cause_ms.load());
        }
    }
    // Frame generation (vk_framegen.h): DLSS-G (vk_dlssg.cpp) or FSR 3 (vk_fsrfg.cpp).
    switch (Vulkan::FrameGen::Current()) {
    case Vulkan::FrameGen::Mode::Off:
        ImGui::TextColored(grey, SotcUi::T("Geração de quadros: desligada", "Frame generation: off"));
        break;
    case Vulkan::FrameGen::Mode::Dlss:
        if (const int fg = SotcStats::ov_fg_state.load(std::memory_order_relaxed); fg == 1) {
            const uint32_t shown = SotcStats::ov_fg_frames.load(std::memory_order_relaxed);
            const uint32_t n = shown ? shown : 2;
            ImGui::TextColored(green, SotcUi::T("DLSS-G x%u: ~%.0f FPS na tela", "DLSS-G x%u: ~%.0f FPS shown"), n, avg_fps * n);
        } else if (fg == 2) {
            ImGui::TextColored(red, "DLSS-G : erreur (statut %#x)",
                               SotcStats::ov_fg_status.load(std::memory_order_relaxed));
        } else {
            ImGui::TextColored(grey, SotcUi::T("DLSS-G: aguardando (sem cena 3D)", "DLSS-G: waiting (no 3D scene)"));
        }
        break;
    case Vulkan::FrameGen::Mode::Fsr:
        if (const int fg = SotcStats::ov_fsr_state.load(std::memory_order_relaxed); fg == 1) {
            ImGui::TextColored(green, SotcUi::T("FSR 3 x2: ~%.0f FPS na tela", "FSR 3 x2: ~%.0f FPS shown"), avg_fps * 2);
        } else if (fg == 2) {
            ImGui::TextColored(red, "FSR 3 : erreur (voir le journal)");
        } else {
            ImGui::TextColored(grey, SotcUi::T("FSR 3: aguardando (sem cena 3D)", "FSR 3: waiting (no 3D scene)"));
        }
        break;
    }
    if (const uint64_t vram = SotcStats::vram_used.load(std::memory_order_relaxed)) {
        ImGui::TextColored(grey, "VRAM %.1f / %.1f Go", vram / 1073741824.0,
                           SotcStats::vram_budget.load(std::memory_order_relaxed) / 1073741824.0);
        ImGui::SameLine();
    }
    ImGui::TextDisabled(SotcUi::T("F7: %s   F8: geração de quadros   F10: esconder", "F7: %s   F8: frame generation   F10: hide"),
                        SotcFpsCap::uncapped ? SotcUi::T("FPS livres", "FPS uncapped") : SotcUi::T("travado em 30", "capped at 30"));
    ImGui::End();
}

void Toast() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                   viewport->WorkPos.y + 24.0f * base_scale),
                            ImGuiCond_Always, ImVec2(0.5f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.7f);
    ImGui::Begin("##toast", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    ImGui::SetWindowFontScale(1.3f);
    ImGui::TextUnformatted(toast_text.c_str());
    ImGui::SetWindowFontScale(1.0f);
    ImGui::End();
}

} // namespace

void Init(const Vulkan::Instance& instance, vk::Format format, u32 image_count) {
    std::scoped_lock lock{imgui_mutex};
    if (initialized) {
        return;
    }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr; // window positions are not kept
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
    io.BackendPlatformName = "bbport";

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.Colors[ImGuiCol_WindowBg].w = 0.92f;

    ImFontConfig font_config;
    font_config.FontDataOwnedByAtlas = false;
    io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(bb_font_ttf),
                                   int(bb_font_ttf_end - bb_font_ttf), 18.0f, &font_config);

    const vk::Instance vk_instance = instance.GetInstance();
    ImGui_ImplVulkan_LoadFunctions(
        instance.ApiVersion(),
        [](const char* name, void* user) {
            return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr(
                *static_cast<const vk::Instance*>(user), name);
        },
        const_cast<vk::Instance*>(&vk_instance));

    const VkFormat color_format = static_cast<VkFormat>(format);
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = instance.ApiVersion();
    info.Instance = vk_instance;
    info.PhysicalDevice = instance.GetPhysicalDevice();
    info.Device = instance.GetDevice();
    info.QueueFamily = instance.GetGraphicsQueueFamilyIndex();
    info.Queue = instance.GetGraphicsQueue();
    info.DescriptorPoolSize = 16;
    info.MinImageCount = std::max(image_count, 2u);
    info.ImageCount = std::max(image_count, 2u);
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.PipelineRenderingCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &color_format,
    };
    if (!ImGui_ImplVulkan_Init(&info)) {
        std::printf("Overlay: ImGui Vulkan backend init failed\n");
        ImGui::DestroyContext();
        return;
    }
    initialized = true;
    std::printf("Overlay: ready\n");
}

bool HandleEvent(const SDL_Event& event) {
    std::scoped_lock lock{imgui_mutex};
    if (!initialized) {
        return false;
    }
    ImGuiIO& io = ImGui::GetIO();
    // sotc: F7 caps the game at 30 FPS or uncaps it (sotc_fps_cap.h).
    // The new state is shown at the next frame (Render).
    if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_F7 && !event.key.repeat) {
        SotcFpsCap::Toggle();
        return true;
    }
    // sotc: F8 switches the frame generation mode (off, DLSS-G, FSR 3; vk_framegen.h).
    // The new mode is shown at the next frame (Render).
    if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_F8 && !event.key.repeat) {
        const Vulkan::FrameGen::Mode mode = Vulkan::FrameGen::Cycle();
        if (mode == Vulkan::FrameGen::Mode::Off &&
            !Vulkan::FrameGen::Available(Vulkan::FrameGen::Mode::Fsr) &&
            !Vulkan::FrameGen::Available(Vulkan::FrameGen::Mode::Dlss)) {
            toast_text = SotcUi::T("Geração de quadros: indisponível nesta placa", "Frame generation: not available on this GPU");
            toast_until_ms.store(NowMs() + 2500, std::memory_order_relaxed);
        }
        return true;
    }
    // sotc: F10 shows or hides the performance overlay.
    if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_F10 && !event.key.repeat) {
        const bool on = !OverlayOn();
        BbSettings::Get().show_fps = false; // the ini setting only gives the initial state
        overlay_on.store(on, std::memory_order_relaxed);
        return true;
    }
    if (list_active && list_result == -1) {
        const int count = static_cast<int>(list_items.size());
        const auto move = [&](int delta) {
            list_confirm = -1;
            if (count) {
                list_selected = (list_selected + delta + count) % count;
                list_scroll = true;
            }
        };
        switch (event.type) {
        case SDL_EVENT_KEY_DOWN:
            switch (event.key.key) {
            case SDLK_UP:
            case SDLK_W:
                move(-1);
                break;
            case SDLK_DOWN:
            case SDLK_S:
                move(1);
                break;
            case SDLK_RETURN:
            case SDLK_KP_ENTER:
            case SDLK_SPACE:
                if (!event.key.repeat && list_confirm >= 0) {
                    list_result = -100 - list_confirm; // delete it (PollSaveList)
                    list_confirm = -1;
                } else if (!event.key.repeat && count) {
                    list_result = list_selected;
                }
                break;
            case SDLK_ESCAPE:
            case SDLK_BACKSPACE:
                if (!event.key.repeat) {
                    if (list_confirm >= 0) {
                        list_confirm = -1;
                    } else {
                        list_result = -2;
                    }
                }
                break;
            case SDLK_DELETE:
                if (!event.key.repeat && list_selected >= list_first_deletable &&
                    list_selected < count) {
                    list_confirm = list_selected;
                }
                break;
            default:
                break;
            }
            return true;
        case SDL_EVENT_KEY_UP:
        case SDL_EVENT_TEXT_INPUT:
            return true;
        case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
            switch (event.gbutton.button) {
            case SDL_GAMEPAD_BUTTON_DPAD_UP:
                move(-1);
                break;
            case SDL_GAMEPAD_BUTTON_DPAD_DOWN:
                move(1);
                break;
            case SDL_GAMEPAD_BUTTON_SOUTH:
                if (list_confirm >= 0) {
                    list_result = -100 - list_confirm;
                    list_confirm = -1;
                } else if (count) {
                    list_result = list_selected;
                }
                break;
            case SDL_GAMEPAD_BUTTON_EAST:
                if (list_confirm >= 0) {
                    list_confirm = -1;
                } else {
                    list_result = -2;
                }
                break;
            case SDL_GAMEPAD_BUTTON_NORTH:
                if (list_selected >= list_first_deletable && list_selected < count) {
                    list_confirm = list_selected;
                }
                break;
            default:
                break;
            }
            return true;
        case SDL_EVENT_GAMEPAD_BUTTON_UP:
            return true;
        case SDL_EVENT_GAMEPAD_AXIS_MOTION:
            if (event.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTY) {
                const bool down = event.gaxis.value > 20000, up = event.gaxis.value < -20000;
                if (down && !list_stick_down) {
                    move(1);
                }
                if (up && !list_stick_up) {
                    move(-1);
                }
                list_stick_down = down;
                list_stick_up = up;
            }
            return true;
        case SDL_EVENT_MOUSE_MOTION: {
            const float density = PixelDensity(event.motion.windowID);
            io.AddMousePosEvent(event.motion.x * density, event.motion.y * density);
            return true;
        }
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (event.button.button == SDL_BUTTON_LEFT) {
                io.AddMouseButtonEvent(0, event.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
            }
            return true;
        case SDL_EVENT_MOUSE_WHEEL:
            io.AddMouseWheelEvent(event.wheel.x, event.wheel.y);
            return true;
        default:
            break;
        }
    }
    return false;
}

bool LoadingShown() {
    // sotc: a second without a game flip. Read from the presenter and the video out threads: the
    // races only move the moment by a frame.
    static std::atomic<u64> seen{~u64{0}};
    static std::atomic<long long> since{0};
    const u64 flips = SotcStats::flip_count.load(std::memory_order_relaxed);
    const long long now = NowMs();
    if (seen.exchange(flips, std::memory_order_relaxed) != flips) {
        since.store(now, std::memory_order_relaxed);
        return false;
    }
    return now - since.load(std::memory_order_relaxed) >= 1000;
}

bool Visible() {
    return initialized && (text_entry_active || list_active || OverlayOn() || ToastShown() ||
                           LoadingShown());
}

bool CapturesInput() {
    return text_entry_active || list_active;
}

bool OpenSaveList(const std::string& title, const std::vector<std::string>& items, int focus,
                  int first_deletable) {
    std::scoped_lock lock{imgui_mutex};
    if (!initialized) {
        return false; // nothing could show it
    }
    list_title = title;
    list_items = items;
    list_icons = std::move(pending_icons); // sotc: the pictures sent just before (or none)
    pending_icons.clear();
    list_selected = items.empty() ? 0 : std::clamp(focus, 0, static_cast<int>(items.size()) - 1);
    list_result = -1;
    list_confirm = -1;
    list_first_deletable = first_deletable < 0 ? static_cast<int>(items.size()) : first_deletable;
    list_scroll = true;
    list_active = true;
    ImGui::GetIO().MouseDrawCursor = true;
    return true;
}

void SetSaveListIcons(std::vector<std::string> paths) {
    std::scoped_lock lock{imgui_mutex};
    pending_icons = std::move(paths);
}

int PollSaveList() {
    std::scoped_lock lock{imgui_mutex};
    if (!list_active || list_result == -1) {
        return -1;
    }
    list_active = false;
    if (initialized) {
        ImGui::GetIO().MouseDrawCursor = false;
    }
    return list_result;
}

void CloseSaveList() {
    std::scoped_lock lock{imgui_mutex};
    list_active = false;
    if (initialized) {
        ImGui::GetIO().MouseDrawCursor = false;
    }
}

void SetTextEntry(bool active, const std::string& prompt, const std::string& text) {
    std::scoped_lock lock{imgui_mutex};
    text_entry_active = active;
    text_entry_prompt = prompt;
    text_entry_text = text;
}

void Render(vk::CommandBuffer cmdbuf, vk::ImageView view, vk::Extent2D extent,
            bool game_frame) {
    // Present interval for the FPS readout (measured also while nothing is drawn), game frames
    // only: a generated frame (FSR 3) is drawn half a frame later.
    const auto now = std::chrono::steady_clock::now();
    static auto last_draw = now;
    const float draw_ms = std::chrono::duration<float, std::milli>(now - last_draw).count();
    last_draw = now;
    if (game_frame) {
        const float ms = std::chrono::duration<float, std::milli>(now - last_present).count();
        last_present = now;
        if (ms > 0.0f && ms < 1000.0f) {
            frame_ms_avg = frame_ms_avg == 0.0f ? ms : frame_ms_avg * 0.95f + ms * 0.05f;
        }
        if (ms > 0.0f && ms < 5000.0f) {
            RecordFrame(ms);
        }
    }
    const float ms = draw_ms;
    // The frame generation mode changed (F8): shown for 2.5 s.
    static Vulkan::FrameGen::Mode shown_mode = Vulkan::FrameGen::Current();
    if (const Vulkan::FrameGen::Mode mode = Vulkan::FrameGen::Current(); mode != shown_mode) {
        shown_mode = mode;
        std::scoped_lock lock{imgui_mutex};
        toast_text = std::string(SotcUi::T("Geração de quadros: ", "Frame generation: ")) + Vulkan::FrameGen::Name(mode);
        toast_until_ms.store(NowMs() + 2500, std::memory_order_relaxed);
    }
    // The 30 FPS cap was switched (F7).
    static bool shown_uncapped = SotcFpsCap::uncapped;
    if (const bool uncapped = SotcFpsCap::uncapped; uncapped != shown_uncapped) {
        shown_uncapped = uncapped;
        std::scoped_lock lock{imgui_mutex};
        toast_text = uncapped ? SotcUi::T("FPS livres", "FPS uncapped") : SotcUi::T("FPS travados em 30", "FPS capped at 30");
        toast_until_ms.store(NowMs() + 2500, std::memory_order_relaxed);
    }
    if (!Visible()) {
        return;
    }
    std::scoped_lock lock{imgui_mutex};
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(float(extent.width), float(extent.height));
    io.DeltaTime = ms > 0.0f && ms < 1000.0f ? ms / 1000.0f : 1.0f / 60.0f;
    // UI scale follows the display height (1080p = 1).
    const float scale = std::max(float(extent.height) / 1080.0f, 0.75f);
    if (std::abs(scale - base_scale) > 0.01f) {
        ImGuiStyle& style = ImGui::GetStyle();
        style.ScaleAllSizes(scale / base_scale);
        style.FontScaleMain = scale;
        base_scale = scale;
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
    const bool loading = LoadingShown() && !list_active && !text_entry_active;
    static bool was_loading = false;
    static long long loading_since = 0;
    if (loading != was_loading) { // sotc: one log line when it appears and when it goes away
        was_loading = loading;
        if (loading) {
            loading_since = NowMs();
            std::fprintf(stderr, "SOTC loading screen: shown\n");
        } else {
            std::fprintf(stderr, "SOTC loading screen: hidden after %.1f s\n",
                         double(NowMs() - loading_since) / 1000.0);
        }
    }
    if (loading) {
        LoadingScreen();
    }
    if (OverlayOn()) {
        FpsCounter();
    }
    if (ToastShown()) {
        Toast();
    }
    if (text_entry_active) {
        TextEntryBox();
    }
    if (list_active) {
        SaveListBox();
    }
    ImGui::Render();

    const vk::RenderingAttachmentInfo attachment{
        .imageView = view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    cmdbuf.beginRendering(vk::RenderingInfo{
        .renderArea = {{0, 0}, extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment,
    });
    {
        // Font atlas uploads submit to the graphics queue themselves.
        std::scoped_lock submit_lock{Vulkan::Scheduler::submit_mutex};
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdbuf);
    }
    cmdbuf.endRendering();
}

} // namespace BbOverlay
