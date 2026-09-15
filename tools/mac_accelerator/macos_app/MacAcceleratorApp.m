#import <Cocoa/Cocoa.h>
#import <Security/Security.h>
#import <sys/stat.h>

@interface AppDelegate : NSObject <NSApplicationDelegate>
@property(nonatomic, strong) NSWindow *window;
@property(nonatomic, strong) NSTextField *repositoryField;
@property(nonatomic, strong) NSTextField *statusLabel;
@property(nonatomic, strong) NSTextView *logView;
@property(nonatomic, strong) NSButton *startButton;
@property(nonatomic, strong) NSButton *stopButton;
@property(nonatomic, strong) NSButton *localTestButton;
@property(nonatomic, strong) NSMutableString *pendingServerOutput;
@property(nonatomic, strong) NSTask *serverTask;
@property(nonatomic, strong) NSPipe *outputPipe;
@property(nonatomic, strong) id performanceActivity;
@property(nonatomic) BOOL userRequestedStop;
@end

@implementation AppDelegate

- (NSString *)inferredRepositoryPath {
  NSURL *location = NSBundle.mainBundle.bundleURL;
  for (NSInteger index = 0; index < 8; index++) {
    location = location.URLByDeletingLastPathComponent;
    NSString *script = [location.path stringByAppendingPathComponent:@"tools/mac_accelerator/run_coreml_server.sh"];
    if ([NSFileManager.defaultManager fileExistsAtPath:script]) return location.path;
  }
  return @"";
}

- (NSString *)keyPath {
  NSString *support = NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory,
                                                           NSUserDomainMask, YES).firstObject;
  return [[support stringByAppendingPathComponent:@"Sunnypilot Mac Accelerator"]
          stringByAppendingPathComponent:@"auth.key"];
}

- (NSTextField *)label:(NSString *)text frame:(NSRect)frame size:(CGFloat)size {
  NSTextField *label = [[NSTextField alloc] initWithFrame:frame];
  label.stringValue = text;
  label.editable = NO;
  label.bezeled = NO;
  label.drawsBackground = NO;
  label.font = [NSFont systemFontOfSize:size];
  return label;
}

- (NSButton *)button:(NSString *)title frame:(NSRect)frame action:(SEL)action {
  NSButton *button = [[NSButton alloc] initWithFrame:frame];
  button.title = title;
  button.bezelStyle = NSBezelStyleRounded;
  button.target = self;
  button.action = action;
  return button;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
  NSString *iconPath = [NSBundle.mainBundle pathForResource:@"AppIcon" ofType:@"png"];
  NSImage *icon = iconPath ? [[NSImage alloc] initWithContentsOfFile:iconPath] : nil;
  if (icon) {
    NSApplication.sharedApplication.applicationIconImage = icon;
  }

  NSRect frame = NSMakeRect(0, 0, 780, 500);
  self.window = [[NSWindow alloc] initWithContentRect:frame
                                            styleMask:(NSWindowStyleMaskTitled |
                                                       NSWindowStyleMaskClosable |
                                                       NSWindowStyleMaskMiniaturizable |
                                                       NSWindowStyleMaskResizable)
                                              backing:NSBackingStoreBuffered defer:NO];
  self.window.title = @"Sunnypilot Mac Accelerator — Bench-only 0.2.2";
  self.window.minSize = NSMakeSize(700, 440);
  [self.window center];
  NSView *view = self.window.contentView;

  NSTextField *title = [self label:@"Sunnypilot Mac Accelerator"
                              frame:NSMakeRect(20, 455, 500, 30) size:22];
  title.font = [NSFont boldSystemFontOfSize:22];
  [view addSubview:title];
  [view addSubview:[self label:@"Core ML + Apple Neural Engine benchmark launcher · Live camera capture disabled"
                              frame:NSMakeRect(20, 430, 730, 22) size:13]];

  NSString *saved = [NSUserDefaults.standardUserDefaults stringForKey:@"repositoryPath"];
  self.repositoryField = [[NSTextField alloc] initWithFrame:NSMakeRect(20, 390, 625, 28)];
  self.repositoryField.stringValue = saved ?: self.inferredRepositoryPath;
  [view addSubview:self.repositoryField];
  [view addSubview:[self button:@"Choose…" frame:NSMakeRect(655, 389, 105, 30)
                              action:@selector(chooseRepository:)]];

  self.statusLabel = [self label:@"●  Stopped" frame:NSMakeRect(20, 350, 260, 26) size:15];
  [view addSubview:self.statusLabel];
  [view addSubview:[self button:@"Copy key path" frame:NSMakeRect(335, 348, 125, 30)
                              action:@selector(copyKeyPath:)]];
  [view addSubview:[self button:@"README" frame:NSMakeRect(465, 348, 90, 30)
                              action:@selector(openReadme:)]];
  self.startButton = [self button:@"Start" frame:NSMakeRect(660, 348, 100, 30)
                                 action:@selector(startServer:)];
  self.stopButton = [self button:@"Stop" frame:NSMakeRect(560, 348, 95, 30)
                                action:@selector(stopServer:)];
  self.stopButton.enabled = NO;
  [view addSubview:self.startButton];
  [view addSubview:self.stopButton];

  NSScrollView *scroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(20, 55, 740, 280)];
  scroll.hasVerticalScroller = YES;
  scroll.borderType = NSBezelBorder;
  self.logView = [[NSTextView alloc] initWithFrame:scroll.bounds];
  self.logView.editable = NO;
  self.logView.font = [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular];
  self.logView.string = @"Bench-only prerelease. Choose a prepared checkout, then Start for a localhost server.\nNo models or Python environment are bundled. README includes setup requirements.\nUSB mode is for synthetic off-road diagnostics only; live camera capture is disabled.";
  scroll.documentView = self.logView;
  [view addSubview:scroll];

  self.localTestButton = [NSButton checkboxWithTitle:@"Mac-only test (localhost, no 3X)" target:nil action:nil];
  self.localTestButton.state = NSControlStateValueOn;
  self.localTestButton.toolTip = @"Default: localhost only. Uncheck solely for synthetic USB bench diagnostics with an off-road 3X.";
  self.localTestButton.frame = NSMakeRect(20, 325, 360, 22);
  [view addSubview:self.localTestButton];
  scroll.frame = NSMakeRect(20, 55, 740, 265);

  NSTextField *warning = [self label:@"Bench-only: live capture disabled. Server status is not driving readiness; no vehicle-control output."
                                  frame:NSMakeRect(20, 18, 740, 24) size:12];
  warning.textColor = NSColor.systemOrangeColor;
  [view addSubview:warning];
  [self.window makeKeyAndOrderFront:nil];
  [NSApp activateIgnoringOtherApps:YES];
}

- (void)setStatus:(NSString *)status running:(BOOL)running {
  self.statusLabel.stringValue = [NSString stringWithFormat:@"%@  %@", running ? @"●" : @"○", status];
  BOOL ready = [status isEqualToString:@"Ready"] || [status isEqualToString:@"Client connected"];
  self.statusLabel.textColor = ready ? NSColor.systemGreenColor : NSColor.labelColor;
  self.startButton.enabled = !running;
  self.stopButton.enabled = running;
  self.localTestButton.enabled = !running;
}

- (void)appendLog:(NSString *)message {
  if (message.length == 0) return;
  NSString *separator = self.logView.string.length ? @"\n" : @"";
  self.logView.string = [self.logView.string stringByAppendingFormat:@"%@%@", separator, message];
  if (self.logView.string.length > 40000) {
    self.logView.string = [self.logView.string substringFromIndex:self.logView.string.length - 40000];
  }
  [self.logView scrollRangeToVisible:NSMakeRange(self.logView.string.length, 0)];
  if ([message hasPrefix:@"listening on "]) [self setStatus:@"Ready" running:YES];
  if ([message hasPrefix:@"verified client connected:"]) [self setStatus:@"Client connected" running:YES];
  if ([message hasPrefix:@"client disconnected:"]) [self setStatus:@"Ready" running:YES];
  if ([message hasPrefix:@"waiting for USB NCM"]) [self setStatus:@"Waiting for USB" running:YES];
  if ([message hasPrefix:@"USB link changed;"]) [self setStatus:@"Reconnecting USB" running:YES];
}

- (void)consumeServerOutput:(NSString *)text {
  if (!text) return;
  [self.pendingServerOutput appendString:text];
  NSRange newline;
  while ((newline = [self.pendingServerOutput rangeOfString:@"\n"]).location != NSNotFound) {
    NSString *line = [self.pendingServerOutput substringToIndex:newline.location];
    [self.pendingServerOutput deleteCharactersInRange:NSMakeRange(0, newline.location + 1)];
    [self appendLog:line];
  }
}

- (BOOL)ensureAuthenticationKey:(NSError **)error {
  NSString *keyPath = self.keyPath;
  NSString *directory = keyPath.stringByDeletingLastPathComponent;
  if (![NSFileManager.defaultManager createDirectoryAtPath:directory withIntermediateDirectories:YES
                                                attributes:@{NSFilePosixPermissions: @0700} error:error]) return NO;
  if (chmod(directory.fileSystemRepresentation, 0700) != 0) {
    if (error) *error = [NSError errorWithDomain:@"MacAccelerator" code:3
      userInfo:@{NSLocalizedDescriptionKey: @"Cannot secure authentication directory"}];
    return NO;
  }
  if ([NSFileManager.defaultManager fileExistsAtPath:keyPath]) {
    NSData *key = [NSData dataWithContentsOfFile:keyPath options:0 error:error];
    if (!key) return NO;
    if (key.length < 32 || chmod(keyPath.fileSystemRepresentation, 0600) != 0) {
      if (error) *error = [NSError errorWithDomain:@"MacAccelerator" code:2
        userInfo:@{NSLocalizedDescriptionKey: @"Authentication key is invalid or cannot be secured"}];
      return NO;
    }
    return YES;
  }
  NSMutableData *key = [NSMutableData dataWithLength:32];
  if (SecRandomCopyBytes(kSecRandomDefault, key.length, key.mutableBytes) != errSecSuccess) {
    if (error) *error = [NSError errorWithDomain:@"MacAccelerator" code:1
                                        userInfo:@{NSLocalizedDescriptionKey: @"Could not generate authentication key"}];
    return NO;
  }
  if (![key writeToFile:keyPath options:NSDataWritingAtomic error:error]) return NO;
  if (chmod(keyPath.fileSystemRepresentation, 0600) != 0) {
    if (error) *error = [NSError errorWithDomain:@"MacAccelerator" code:4
      userInfo:@{NSLocalizedDescriptionKey: @"Cannot secure new authentication key"}];
    return NO;
  }
  [self appendLog:@"Created a private 32-byte authentication key"];
  return YES;
}

- (void)chooseRepository:(id)sender {
  NSOpenPanel *panel = NSOpenPanel.openPanel;
  panel.title = @"Choose the sunnypilot checkout";
  panel.canChooseDirectories = YES;
  panel.canChooseFiles = NO;
  panel.allowsMultipleSelection = NO;
  if ([panel runModal] == NSModalResponseOK) {
    self.repositoryField.stringValue = panel.URL.path;
    [NSUserDefaults.standardUserDefaults setObject:panel.URL.path forKey:@"repositoryPath"];
    [self appendLog:[@"Selected repository: " stringByAppendingString:panel.URL.path]];
  }
}

- (void)startServer:(id)sender {
  if (self.serverTask.running) return;
  NSString *root = self.repositoryField.stringValue.stringByStandardizingPath;
  NSArray<NSString *> *required = @[
    [root stringByAppendingPathComponent:@"tools/mac_accelerator/run_coreml_server.sh"],
    [root stringByAppendingPathComponent:@".coreml-venv/bin/python"],
    [root stringByAppendingPathComponent:@"tools/mac_accelerator/artifacts/big_driving.mlpackage"],
    [root stringByAppendingPathComponent:@"tools/mac_accelerator/usb_ncm_supervisor.py"],
    [root stringByAppendingPathComponent:@"openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"],
    [root stringByAppendingPathComponent:@"openpilot/selfdrive/modeld/models/big_driving_supercombo_metadata.pkl"],
  ];
  for (NSString *path in required) {
    if (![NSFileManager.defaultManager fileExistsAtPath:path]) {
      [self setStatus:@"Setup required" running:NO];
      [self appendLog:[@"Missing: " stringByAppendingString:path]];
      return;
    }
  }
  NSError *error = nil;
  if (![self ensureAuthenticationKey:&error]) {
    [self setStatus:@"Key error" running:NO];
    [self appendLog:error.localizedDescription];
    return;
  }

  NSTask *task = [[NSTask alloc] init];
  task.executableURL = [NSURL fileURLWithPath:@"/usr/bin/caffeinate"];
  task.arguments = @[@"-dimsu", required[0]];
  task.qualityOfService = NSQualityOfServiceUserInteractive;
  task.currentDirectoryURL = [NSURL fileURLWithPath:root];
  NSMutableDictionary *environment = NSProcessInfo.processInfo.environment.mutableCopy;
  NSString *existingPath = environment[@"PATH"] ?: @"/usr/bin:/bin:/usr/sbin:/sbin";
  environment[@"PATH"] = [NSString stringWithFormat:@"/opt/homebrew/bin:/usr/local/bin:%@", existingPath];
  environment[@"AUTH_KEY_FILE"] = self.keyPath;
  environment[@"PYTHONUNBUFFERED"] = @"1";
  environment[@"VECLIB_MAXIMUM_THREADS"] = @"1";
  environment[@"OPENBLAS_NUM_THREADS"] = @"1";
  environment[@"MAC_ACCELERATOR_LOCAL_TEST"] = self.localTestButton.state == NSControlStateValueOn ? @"1" : @"0";
  task.environment = environment;
  NSPipe *pipe = NSPipe.pipe;
  task.standardOutput = pipe;
  task.standardError = pipe;
  __weak typeof(self) weakSelf = self;
  self.pendingServerOutput = [NSMutableString string];
  pipe.fileHandleForReading.readabilityHandler = ^(NSFileHandle *handle) {
    NSData *data = handle.availableData;
    if (data.length == 0) return;
    NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf consumeServerOutput:text]; });
  };
  task.terminationHandler = ^(NSTask *finished) {
    dispatch_async(dispatch_get_main_queue(), ^{
      BOOL cleanStop = weakSelf.userRequestedStop;
      weakSelf.userRequestedStop = NO;
      weakSelf.outputPipe.fileHandleForReading.readabilityHandler = nil;
      weakSelf.serverTask = nil;
      weakSelf.outputPipe = nil;
      if (weakSelf.performanceActivity) {
        [NSProcessInfo.processInfo endActivity:weakSelf.performanceActivity];
        weakSelf.performanceActivity = nil;
      }
      [weakSelf setStatus:(finished.terminationStatus == 0 || cleanStop ? @"Stopped" : @"Stopped with error") running:NO];
      [weakSelf appendLog:[NSString stringWithFormat:@"Server exited with status %d", finished.terminationStatus]];
    });
  };
  self.performanceActivity = [NSProcessInfo.processInfo
    beginActivityWithOptions:(NSActivityUserInitiatedAllowingIdleSystemSleep | NSActivityLatencyCritical)
                      reason:@"Sunnypilot 20 Hz Core ML inference"];
  if (![task launchAndReturnError:&error]) {
    [NSProcessInfo.processInfo endActivity:self.performanceActivity];
    self.performanceActivity = nil;
    [self setStatus:@"Start failed" running:NO];
    [self appendLog:error.localizedDescription];
    return;
  }
  self.serverTask = task;
  self.outputPipe = pipe;
  self.userRequestedStop = NO;
  [self setStatus:@"Starting" running:YES];
  [self appendLog:@"Starting Core ML/ANE worker in dedicated awake mode"];
}

- (void)stopServer:(id)sender {
  if (self.serverTask.running) {
    self.userRequestedStop = YES;
    [self setStatus:@"Stopping" running:YES];
    [self.serverTask terminate];
  }
}

- (void)copyKeyPath:(id)sender {
  [NSPasteboard.generalPasteboard clearContents];
  [NSPasteboard.generalPasteboard setString:self.keyPath forType:NSPasteboardTypeString];
  [self appendLog:@"Authentication key path copied"];
}

- (void)openReadme:(id)sender {
  NSString *path = [self.repositoryField.stringValue.stringByStandardizingPath
                    stringByAppendingPathComponent:@"tools/mac_accelerator/README.md"];
  if (![NSFileManager.defaultManager fileExistsAtPath:path]) {
    path = [NSBundle.mainBundle pathForResource:@"BENCH_SETUP" ofType:@"txt"];
  }
  if (!path) return;
  [NSWorkspace.sharedWorkspace openURL:[NSURL fileURLWithPath:path]];
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
  if (self.serverTask.running) [self.serverTask terminate];
  return NSTerminateNow;
}

@end

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    NSApplication *application = NSApplication.sharedApplication;
    AppDelegate *delegate = [[AppDelegate alloc] init];
    application.delegate = delegate;
    [application run];
  }
  return 0;
}
