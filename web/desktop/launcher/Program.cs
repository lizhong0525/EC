using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class Program
{
    [STAThread]
    private static int Main(string[] args)
    {
        bool checkOnly = args.Length == 1 && args[0] == "--check";
        if (args.Length != 0 && !checkOnly) return 1;

        string root = AppDomain.CurrentDomain.BaseDirectory;
        string electron = Path.Combine(root, "web", "node_modules", "electron", "dist", "electron.exe");
        string main = Path.Combine(root, "web", "desktop", "main.cjs");
        string[] required = {
            electron, main,
            Path.Combine(root, "web", "desktop", "preload.cjs"),
            Path.Combine(root, "web", "dist", "index.html"),
            Path.Combine(root, ".research-venv", "Scripts", "python.exe"),
            Path.Combine(root, "backend", "src", "ecocompanion", "pet", "api.py")
        };
        foreach (string file in required)
        {
            if (!File.Exists(file))
            {
                if (!checkOnly)
                    ShowError("A required application file is missing:\n\n" + file
                        + "\n\nKeep EcoCompanion.exe in the project folder. "
                        + "This preview uses the project's existing Python and Electron runtimes.");
                return 2;
            }
        }
        if (checkOnly) return 0;

        try
        {
            var start = new ProcessStartInfo {
                FileName = electron,
                Arguments = "\"" + main + "\"",
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden
            };
            start.EnvironmentVariables.Remove("ELECTRON_RUN_AS_NODE");
            using (Process process = Process.Start(start))
            {
                if (process == null) throw new InvalidOperationException("Electron did not start.");
            }
            return 0;
        }
        catch (Exception error)
        {
            ShowError("Unable to start the desktop companion.\n\n" + error.Message);
            return 3;
        }
    }

    private static void ShowError(string text)
    {
        Application.EnableVisualStyles();
        MessageBox.Show(text, "EcoCompanion", MessageBoxButtons.OK, MessageBoxIcon.Error);
    }
}
